// SPDX-License-Identifier: AGPL-3.0-only

//! Single-step classifier, routing, and structured decision scoring (`POST /v1/decisions`)
//! plus [`logprob_token_ids`] candidate token resolution.
//!
//! Supports all three industry wire contracts on one endpoint:
//! 1. **Flare / vLLM Decisions** (`questions` is a JSON Array, `input` context):
//!    `predicate`, `choice`, and `score` question objects.
//! 2. **Eider / Decisive Dory** (`questions` is a JSON Object/Map, `state` context):
//!    `noul`, `choice` (`criteria` map + `concentration`), and `score` (`criteria` array + `legend`).
//! 3. **Flat Choice Scoring** (`choices: ["...", ...]` at top level, with optional
//!    `resolve_only: true` or pre-computed `token_logprobs` map).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::compact::openai_error_response;
use super::inference_impl::try_admit_scheduler;
use super::inference_types::{InferenceRequest, TokenLogprobs};
use crate::AppState;
use crate::main_modules::model_host::CurrentModel;
use crate::tokenizer::ChatTokenizer;

/// Maximum number of questions evaluated in a single `/v1/decisions` request.
const MAX_DECISION_QUESTIONS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecisionError {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
}

impl DecisionError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }

    fn from_status(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

/// Resolved token IDs for one candidate choice string.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChoiceTokenIds {
    pub choice: String,
    /// Full token sequence of the bare choice string.
    pub token_ids: Vec<u32>,
    /// De-duplicated first-step token IDs covering both bare (`"yes"`) and
    /// space-prefixed (`" yes"`) BPE surface forms.
    pub candidate_first_tokens: Vec<u32>,
}

/// Resolve each candidate choice string to its token ID sequence and first-step
/// BPE surface token IDs.
///
/// # Errors
/// Returns a human-readable error if `choices` is empty, any choice is empty,
/// or tokenization produces zero tokens.
pub fn logprob_token_ids(
    tokenizer: &ChatTokenizer,
    choices: &[String],
) -> Result<Vec<ChoiceTokenIds>, String> {
    if choices.is_empty() {
        return Err("choices must not be empty".to_string());
    }
    let mut out = Vec::with_capacity(choices.len());
    for choice in choices {
        if choice.is_empty() {
            return Err("choice strings must not be empty".to_string());
        }
        let bare_ids = tokenizer
            .encode(choice)
            .map_err(|e| format!("failed to tokenize choice {choice:?}: {e}"))?;
        let Some(&primary) = bare_ids.first() else {
            return Err(format!("choice {choice:?} tokenized to zero tokens"));
        };
        let mut first_tokens = vec![primary];
        if !choice.starts_with(char::is_whitespace) {
            let spaced = format!(" {choice}");
            if let Ok(spaced_ids) = tokenizer.encode(&spaced)
                && let Some(&sp_first) = spaced_ids.first()
                && !first_tokens.contains(&sp_first)
            {
                first_tokens.push(sp_first);
            }
        }
        out.push(ChoiceTokenIds {
            choice: choice.clone(),
            token_ids: bare_ids,
            candidate_first_tokens: first_tokens,
        });
    }
    Ok(out)
}

#[must_use]
pub(crate) fn first_tokens_are_disjoint(resolved: &[ChoiceTokenIds]) -> bool {
    let mut seen = BTreeSet::new();
    for item in resolved {
        for &tok in &item.candidate_first_tokens {
            if !seen.insert(tok) {
                return false;
            }
        }
    }
    true
}

pub(crate) fn logsumexp(xs: &[f64]) -> f64 {
    let Some(&max_x) = xs
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return f64::NEG_INFINITY;
    };
    if !max_x.is_finite() {
        return max_x;
    }
    let sum: f64 = xs.iter().map(|&x| (x - max_x).exp()).sum();
    max_x + sum.ln()
}

pub(crate) fn softmax(log_scores: &[f64]) -> Vec<f64> {
    if log_scores.is_empty() {
        return Vec::new();
    }
    let z = logsumexp(log_scores);
    if !z.is_finite() {
        let uniform = 1.0 / (log_scores.len() as f64);
        return vec![uniform; log_scores.len()];
    }
    log_scores.iter().map(|&lp| (lp - z).exp()).collect()
}

/// Normalized Shannon concentration `1 - H(p) / ln(K)` in `[0.0, 1.0]`.
#[must_use]
pub(crate) fn shannon_concentration(probs: &[f64]) -> f64 {
    if probs.len() <= 1 {
        return 1.0;
    }
    let max_entropy = (probs.len() as f64).ln();
    if max_entropy <= 0.0 {
        return 1.0;
    }
    let entropy: f64 = probs
        .iter()
        .copied()
        .filter(|&p| p > 0.0 && p.is_finite())
        .map(|p| -p * p.ln())
        .sum();
    (1.0 - (entropy / max_entropy)).clamp(0.0, 1.0)
}

#[derive(Debug, Clone)]
pub(crate) struct ScoredChoices {
    pub raw_logprobs: Vec<f64>,
    pub probabilities: Vec<f64>,
    pub best_index: usize,
    pub confidence: f64,
    pub concentration: f64,
}

pub(crate) fn score_candidate_groups(
    lp_map: &BTreeMap<u32, f64>,
    floor_lp: f64,
    groups: &[Vec<u32>],
) -> ScoredChoices {
    let mut raw_logprobs = Vec::with_capacity(groups.len());
    for group in groups {
        let mut matched = Vec::new();
        for tok in group {
            if let Some(&lp) = lp_map.get(tok) {
                matched.push(lp);
            }
        }
        let group_lp = if matched.is_empty() {
            floor_lp
        } else {
            logsumexp(&matched)
        };
        raw_logprobs.push(group_lp);
    }
    let probabilities = softmax(&raw_logprobs);
    let best_index = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx);
    let confidence = raw_logprobs
        .get(best_index)
        .copied()
        .map_or(0.0, |lp| lp.exp().clamp(0.0, 1.0));
    let concentration = shannon_concentration(&probabilities);
    ScoredChoices {
        raw_logprobs,
        probabilities,
        best_index,
        confidence,
        concentration,
    }
}

pub(crate) fn score_boolean_predicate(
    tokenizer: &ChatTokenizer,
    lp_map: &BTreeMap<u32, f64>,
    floor_lp: f64,
) -> f64 {
    let pos_words: Vec<String> = ["true", "True", "yes", "Yes", "1"]
        .into_iter()
        .map(String::from)
        .collect();
    let neg_words: Vec<String> = ["false", "False", "no", "No", "0"]
        .into_iter()
        .map(String::from)
        .collect();

    let collect_ids = |words: &[String]| -> BTreeSet<u32> {
        let mut ids = BTreeSet::new();
        for w in words {
            if let Ok(enc) = tokenizer.encode(w)
                && let Some(&id) = enc.first()
            {
                ids.insert(id);
            }
            let sp = format!(" {w}");
            if let Ok(enc) = tokenizer.encode(&sp)
                && let Some(&id) = enc.first()
            {
                ids.insert(id);
            }
        }
        ids
    };

    let pos_set = collect_ids(&pos_words);
    let neg_set = collect_ids(&neg_words);
    let pos_only: Vec<u32> = pos_set.difference(&neg_set).copied().collect();
    let neg_only: Vec<u32> = neg_set.difference(&pos_set).copied().collect();
    if pos_only.is_empty() || neg_only.is_empty() {
        return 0.5;
    }
    let scored = score_candidate_groups(lp_map, floor_lp, &[pos_only, neg_only]);
    scored.probabilities.first().copied().unwrap_or(0.5)
}

fn logprob_map_from_step(step: &TokenLogprobs) -> (BTreeMap<u32, f64>, f64) {
    let mut map = BTreeMap::new();
    let mut min_lp = f64::from(step.logprob);
    map.insert(step.token_id, f64::from(step.logprob));
    for &(tid, lp) in &step.top {
        let lp64 = f64::from(lp);
        if lp64 < min_lp {
            min_lp = lp64;
        }
        map.entry(tid).or_insert(lp64);
    }
    let floor = if min_lp.is_finite() {
        min_lp - 10.0
    } else {
        -30.0
    };
    (map, floor)
}

fn parse_supplied_logprobs(raw: &Map<String, Value>) -> Result<(BTreeMap<u32, f64>, f64), String> {
    let mut map = BTreeMap::new();
    let mut min_lp = 0.0f64;
    let mut any = false;
    for (k, v) in raw {
        let tid = k
            .parse::<u32>()
            .map_err(|_| format!("token_logprobs key {k:?} is not a valid u32 token ID"))?;
        let lp = v
            .as_f64()
            .ok_or_else(|| format!("token_logprobs[{k:?}] must be a finite number"))?;
        if !lp.is_finite() {
            return Err(format!("token_logprobs[{k:?}] must be finite"));
        }
        if !any || lp < min_lp {
            min_lp = lp;
            any = true;
        }
        map.insert(tid, lp);
    }
    let floor = if any { min_lp - 10.0 } else { -30.0 };
    Ok((map, floor))
}

fn format_context_text(body: &Map<String, Value>) -> Result<Option<Vec<Value>>, String> {
    if let Some(msgs) = body.get("messages").and_then(Value::as_array) {
        if msgs.is_empty() {
            return Err("messages must not be empty".to_string());
        }
        return Ok(Some(msgs.clone()));
    }
    for key in ["input", "state", "prompt"] {
        if let Some(val) = body.get(key) {
            if let Some(arr) = val.as_array()
                && arr.iter().all(|m| m.get("role").is_some())
                && !arr.is_empty()
            {
                return Ok(Some(arr.clone()));
            }
            let text = match val {
                Value::String(s) => s.clone(),
                Value::Null => continue,
                other => serde_json::to_string_pretty(other).unwrap_or_default(),
            };
            if !text.trim().is_empty() {
                return Ok(Some(vec![serde_json::json!({
                    "role": "user",
                    "content": text,
                })]));
            }
        }
    }
    Ok(None)
}

fn append_question_instruction(base_messages: &[Value], instruction: &str) -> Vec<Value> {
    if instruction.trim().is_empty() {
        return base_messages.to_vec();
    }
    let mut msgs = base_messages.to_vec();
    if let Some(last) = msgs.last_mut()
        && last.get("role").and_then(Value::as_str) == Some("user")
        && let Some(content) = last.get("content").and_then(Value::as_str)
    {
        last["content"] = Value::String(format!("{content}\n\n{instruction}"));
        return msgs;
    }
    msgs.push(serde_json::json!({
        "role": "user",
        "content": instruction,
    }));
    msgs
}

async fn run_single_step_logprobs(
    state: &Arc<AppState>,
    messages: &[Value],
) -> Result<(BTreeMap<u32, f64>, f64), DecisionError> {
    let prompt_tokens = if state.dsv41 {
        let wire = serde_json::json!({
            "messages": messages,
            "enable_thinking": false,
        });
        let rendered = crate::dsv41::request::render_request(&wire)
            .map_err(|e| DecisionError::bad_request(e.message))?;
        super::dsv41::escaped_prompt_ids(&state.tokenizer, &wire, &rendered)
            .map_err(|e| DecisionError::bad_request(format!("tokenization failed: {e}")))?
    } else {
        state
            .tokenizer
            .apply_chat_template_openai_with_options(
                messages,
                None,
                false,
                state.behavior.disable_tool_steering,
                Some("none"),
                Some(false),
            )
            .map_err(|e| {
                DecisionError::bad_request(format!("chat template rendering failed: {e}"))
            })?
    };

    if prompt_tokens.len() >= state.max_seq_len {
        return Err(DecisionError::bad_request(format!(
            "prompt length {} exceeds max_seq_len {}",
            prompt_tokens.len(),
            state.max_seq_len
        )));
    }

    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let timeout_secs = if state.request_timeout > 0 {
        u64::from(state.request_timeout)
    } else {
        60
    };
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();

    let infer_req = InferenceRequest::Blocking {
        prompt_tokens,
        session_hash,
        image_pixels: Vec::new(),
        max_tokens: 1,
        min_tokens: 0,
        ignore_eos: true,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        lz_penalty: 0.0,
        logit_bias: Vec::new(),
        stop_tokens: Vec::new(),
        enable_thinking: false,
        thinking_budget: Some(0),
        require_tool_call: false,
        suppress_tool_call: true,
        disable_mtp: true,
        grammar_spec: None,
        seed: Some(0),
        top_logprobs: Some(255),
        timeout_at: Some(deadline),
        decode_timeout_at: Some(deadline),
        intervention_ledger: None,
        think_nudge: None,
        response_tx,
    };

    try_admit_scheduler(&state.request_tx, infer_req)
        .map_err(|(status, msg)| DecisionError::from_status(status, msg))?;
    let resp = response_rx
        .await
        .map_err(|_| DecisionError::internal("scheduler dropped decision response"))?
        .map_err(|e| DecisionError::internal(format!("inference error: {e}")))?;
    let first_lp = resp.logprobs.first().ok_or_else(|| {
        DecisionError::internal("scheduler returned no logprobs for single-step decision")
    })?;
    Ok(logprob_map_from_step(first_lp))
}

#[derive(Debug, Clone)]
struct OptionItem {
    label: String,
    description: Option<String>,
}

fn parse_flare_options(arr: &[Value], key_name: &str) -> Result<Vec<OptionItem>, String> {
    if arr.is_empty() {
        return Err(format!("{key_name} must not be empty"));
    }
    let mut items = Vec::with_capacity(arr.len());
    for entry in arr {
        match entry {
            Value::String(s) if !s.is_empty() => items.push(OptionItem {
                label: s.clone(),
                description: None,
            }),
            Value::Object(obj) => {
                let label = obj
                    .get("value")
                    .or_else(|| obj.get("label"))
                    .or_else(|| obj.get("name"))
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| format!("each {key_name} object requires a non-empty string"))?;
                let description = obj
                    .get("description")
                    .and_then(Value::as_str)
                    .map(String::from);
                items.push(OptionItem {
                    label: label.to_string(),
                    description,
                });
            }
            _ => return Err(format!("invalid entry in {key_name}")),
        }
    }
    Ok(items)
}

async fn evaluate_options(
    state: Option<&Arc<AppState>>,
    tokenizer: &ChatTokenizer,
    base_messages: &[Value],
    question_text: &str,
    items: &[OptionItem],
    offline: Option<&(BTreeMap<u32, f64>, f64)>,
) -> Result<ScoredChoices, DecisionError> {
    let labels: Vec<String> = items.iter().map(|i| i.label.clone()).collect();
    let direct = logprob_token_ids(tokenizer, &labels).map_err(DecisionError::bad_request)?;
    let use_direct = offline.is_some() || first_tokens_are_disjoint(&direct);

    if use_direct {
        let mut lines = Vec::new();
        if !question_text.is_empty() {
            lines.push(format!("Question: {question_text}"));
        }
        lines.push("Valid options (reply with the exact option token only):".to_string());
        for item in items {
            match &item.description {
                Some(desc) if !desc.is_empty() => lines.push(format!("- {}: {desc}", item.label)),
                _ => lines.push(format!("- {}", item.label)),
            }
        }
        let groups: Vec<Vec<u32>> = direct
            .into_iter()
            .map(|c| c.candidate_first_tokens)
            .collect();
        if let Some((lp_map, floor)) = offline {
            return Ok(score_candidate_groups(lp_map, *floor, &groups));
        }
        let live_state = state.ok_or_else(|| DecisionError::internal("no live model state"))?;
        let msgs = append_question_instruction(base_messages, &lines.join("\n"));
        let (lp_map, floor) = run_single_step_logprobs(live_state, &msgs).await?;
        Ok(score_candidate_groups(&lp_map, floor, &groups))
    } else {
        // Shared first-token prefix fallback: assign 1-based numeric keys so a
        // single decode step cleanly disambiguates arbitrary multi-word labels.
        let index_keys: Vec<String> = (1..=items.len()).map(|i| i.to_string()).collect();
        let indexed =
            logprob_token_ids(tokenizer, &index_keys).map_err(DecisionError::bad_request)?;
        let mut lines = Vec::new();
        if !question_text.is_empty() {
            lines.push(format!("Question: {question_text}"));
        }
        lines.push("Reply with ONLY the option number:".to_string());
        for (idx, item) in items.iter().enumerate() {
            match &item.description {
                Some(desc) if !desc.is_empty() => {
                    lines.push(format!("{}: {} ({desc})", idx + 1, item.label))
                }
                _ => lines.push(format!("{}: {}", idx + 1, item.label)),
            }
        }
        let groups: Vec<Vec<u32>> = indexed
            .into_iter()
            .map(|c| c.candidate_first_tokens)
            .collect();
        let live_state = state.ok_or_else(|| DecisionError::internal("no live model state"))?;
        let msgs = append_question_instruction(base_messages, &lines.join("\n"));
        let (lp_map, floor) = run_single_step_logprobs(live_state, &msgs).await?;
        Ok(score_candidate_groups(&lp_map, floor, &groups))
    }
}

pub(crate) async fn handle_flare_questions(
    state: Option<&Arc<AppState>>,
    tokenizer: &ChatTokenizer,
    model_name: &str,
    base_messages: &[Value],
    questions: &[Value],
    offline: Option<&(BTreeMap<u32, f64>, f64)>,
) -> Result<Value, DecisionError> {
    if questions.is_empty() {
        return Err(DecisionError::bad_request(
            "questions array must not be empty",
        ));
    }
    if questions.len() > MAX_DECISION_QUESTIONS {
        return Err(DecisionError::bad_request(format!(
            "too many questions: {} exceeds maximum of {MAX_DECISION_QUESTIONS}",
            questions.len()
        )));
    }
    let mut results = Vec::with_capacity(questions.len());
    for q in questions {
        let obj = q.as_object().ok_or_else(|| {
            DecisionError::bad_request("each question in Flare mode must be a JSON object")
        })?;
        let q_type = obj
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| DecisionError::bad_request("question is missing string field 'type'"))?;
        let name = obj.get("name").and_then(Value::as_str).unwrap_or("");
        let prompt_hint = obj
            .get("question")
            .or_else(|| obj.get("description"))
            .or_else(|| obj.get("prompt"))
            .and_then(Value::as_str)
            .unwrap_or(name);

        let entry = match q_type {
            "predicate" => {
                let p_true = if let Some((lp_map, floor)) = offline {
                    score_boolean_predicate(tokenizer, lp_map, *floor)
                } else {
                    let live_state =
                        state.ok_or_else(|| DecisionError::internal("no live model state"))?;
                    let instr = format!(
                        "Question: {prompt_hint}\nAnswer with a single token: true or false."
                    );
                    let msgs = append_question_instruction(base_messages, &instr);
                    let (lp_map, floor) = run_single_step_logprobs(live_state, &msgs).await?;
                    score_boolean_predicate(tokenizer, &lp_map, floor)
                };
                serde_json::json!({
                    "type": "predicate",
                    "name": name,
                    "probability": p_true,
                })
            }
            "choice" => {
                let arr = obj
                    .get("choices")
                    .or_else(|| obj.get("options"))
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        DecisionError::bad_request("choice question requires 'choices' array")
                    })?;
                let items =
                    parse_flare_options(arr, "choices").map_err(DecisionError::bad_request)?;
                let scored = evaluate_options(
                    state,
                    tokenizer,
                    base_messages,
                    prompt_hint,
                    &items,
                    offline,
                )
                .await?;
                let best = items[scored.best_index].label.clone();
                serde_json::json!({
                    "type": "choice",
                    "name": name,
                    "choice": best,
                    "probabilities": scored.probabilities,
                    "confidence": scored.confidence,
                })
            }
            "score" => {
                let arr = obj
                    .get("levels")
                    .or_else(|| obj.get("criteria"))
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        DecisionError::bad_request("score question requires 'levels' array")
                    })?;
                let items =
                    parse_flare_options(arr, "levels").map_err(DecisionError::bad_request)?;
                let scored = evaluate_options(
                    state,
                    tokenizer,
                    base_messages,
                    prompt_hint,
                    &items,
                    offline,
                )
                .await?;
                let expected_idx: f64 = scored
                    .probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (i as f64) * p)
                    .sum();
                serde_json::json!({
                    "type": "score",
                    "name": name,
                    "score": expected_idx,
                    "probabilities": scored.probabilities,
                    "confidence": scored.confidence,
                })
            }
            other => {
                return Err(DecisionError::bad_request(format!(
                    "unsupported Flare question type {other:?}"
                )));
            }
        };
        results.push(entry);
    }

    Ok(serde_json::json!({
        "object": "decisions",
        "model": model_name,
        "decisions": results,
        "results": results,
    }))
}

pub(crate) async fn handle_eider_questions(
    state: Option<&Arc<AppState>>,
    tokenizer: &ChatTokenizer,
    model_name: &str,
    base_messages: &[Value],
    questions: &Map<String, Value>,
    offline: Option<&(BTreeMap<u32, f64>, f64)>,
) -> Result<Value, DecisionError> {
    if questions.is_empty() {
        return Err(DecisionError::bad_request(
            "questions map must not be empty",
        ));
    }
    if questions.len() > MAX_DECISION_QUESTIONS {
        return Err(DecisionError::bad_request(format!(
            "too many questions: {} exceeds maximum of {MAX_DECISION_QUESTIONS}",
            questions.len()
        )));
    }
    let mut answers = Map::new();
    for (q_key, q_val) in questions {
        let obj = q_val.as_object().ok_or_else(|| {
            DecisionError::bad_request(format!("question {q_key:?} must be a JSON object"))
        })?;
        let q_type = obj.get("type").and_then(Value::as_str).ok_or_else(|| {
            DecisionError::bad_request(format!("question {q_key:?} is missing string field 'type'"))
        })?;
        let prompt_hint = obj
            .get("question")
            .or_else(|| obj.get("description"))
            .or_else(|| obj.get("prompt"))
            .and_then(Value::as_str)
            .unwrap_or(q_key.as_str());

        let ans = match q_type {
            "noul" | "predicate" => {
                let p_true = if let Some((lp_map, floor)) = offline {
                    score_boolean_predicate(tokenizer, lp_map, *floor)
                } else {
                    let live_state =
                        state.ok_or_else(|| DecisionError::internal("no live model state"))?;
                    let instr = format!(
                        "Question: {prompt_hint}\nAnswer with a single token: true or false."
                    );
                    let msgs = append_question_instruction(base_messages, &instr);
                    let (lp_map, floor) = run_single_step_logprobs(live_state, &msgs).await?;
                    score_boolean_predicate(tokenizer, &lp_map, floor)
                };
                serde_json::json!({
                    "type": "noul",
                    "noul": p_true,
                })
            }
            "choice" => {
                let items = match obj.get("criteria").or_else(|| obj.get("choices")) {
                    Some(Value::Object(crit_map)) if !crit_map.is_empty() => crit_map
                        .iter()
                        .map(|(k, v)| OptionItem {
                            label: k.clone(),
                            description: v.as_str().map(String::from),
                        })
                        .collect::<Vec<_>>(),
                    Some(Value::Array(arr)) => {
                        parse_flare_options(arr, "criteria").map_err(DecisionError::bad_request)?
                    }
                    _ => {
                        return Err(DecisionError::bad_request(format!(
                            "Eider choice question {q_key:?} requires non-empty 'criteria' object or array"
                        )));
                    }
                };
                let scored = evaluate_options(
                    state,
                    tokenizer,
                    base_messages,
                    prompt_hint,
                    &items,
                    offline,
                )
                .await?;
                let best = items[scored.best_index].label.clone();
                let mut prob_map = Map::new();
                for (item, &p) in items.iter().zip(&scored.probabilities) {
                    prob_map.insert(item.label.clone(), Value::from(p));
                }
                serde_json::json!({
                    "type": "choice",
                    "choice": best,
                    "probabilities": prob_map,
                    "concentration": scored.concentration,
                })
            }
            "score" => {
                let arr = obj
                    .get("criteria")
                    .or_else(|| obj.get("levels"))
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        DecisionError::bad_request(format!(
                            "Eider score question {q_key:?} requires 'criteria' array"
                        ))
                    })?;
                let items =
                    parse_flare_options(arr, "criteria").map_err(DecisionError::bad_request)?;
                let scored = evaluate_options(
                    state,
                    tokenizer,
                    base_messages,
                    prompt_hint,
                    &items,
                    offline,
                )
                .await?;
                let expected: f64 = scored
                    .probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (i as f64) * p)
                    .sum();
                let mut legend = Map::new();
                let mut prob_map = Map::new();
                for (i, (item, &p)) in items.iter().zip(&scored.probabilities).enumerate() {
                    let idx_key = i.to_string();
                    let desc = item.description.as_ref().unwrap_or(&item.label).clone();
                    legend.insert(idx_key.clone(), Value::String(desc));
                    prob_map.insert(item.label.clone(), Value::from(p));
                }
                serde_json::json!({
                    "type": "score",
                    "score": expected,
                    "legend": legend,
                    "probabilities": prob_map,
                    "concentration": scored.concentration,
                })
            }
            other => {
                return Err(DecisionError::bad_request(format!(
                    "unsupported Eider question type {other:?} in {q_key:?}"
                )));
            }
        };
        answers.insert(q_key.clone(), ans);
    }

    Ok(serde_json::json!({
        "object": "decisions",
        "model": model_name,
        "decisions": answers,
        "answers": answers,
        "questions": answers,
    }))
}

#[derive(Debug, Deserialize)]
struct FlatDecisionFields {
    choices: Vec<String>,
    #[serde(default)]
    resolve_only: bool,
}

/// `POST /v1/decisions` — single-step classifier/choice scoring across Flare,
/// Eider (Decisive Dory), and flat `choices` request shapes.
pub async fn decisions_endpoint(
    CurrentModel(state): CurrentModel,
    body: axum::body::Bytes,
) -> Response {
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    let Some(obj) = parsed.as_object() else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "Request body must be a JSON object".to_string(),
        );
    };

    let offline = match obj.get("token_logprobs") {
        Some(Value::Object(raw_lps)) => match parse_supplied_logprobs(raw_lps) {
            Ok(pair) => Some(pair),
            Err(msg) => return openai_error_response(StatusCode::BAD_REQUEST, msg),
        },
        Some(_) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                "token_logprobs must be a JSON object mapping token IDs to logprobs".to_string(),
            );
        }
        None => None,
    };

    if let Some(questions_val) = obj.get("questions") {
        let base_messages = match format_context_text(obj) {
            Ok(Some(msgs)) => msgs,
            Ok(None) if offline.is_some() => Vec::new(),
            Ok(None) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    "Decision request requires 'input', 'state', 'messages', or 'prompt'"
                        .to_string(),
                );
            }
            Err(msg) => return openai_error_response(StatusCode::BAD_REQUEST, msg),
        };

        let out = match questions_val {
            Value::Array(arr) => {
                handle_flare_questions(
                    Some(&state),
                    &state.tokenizer,
                    &state.model_name,
                    &base_messages,
                    arr,
                    offline.as_ref(),
                )
                .await
            }
            Value::Object(map) => {
                handle_eider_questions(
                    Some(&state),
                    &state.tokenizer,
                    &state.model_name,
                    &base_messages,
                    map,
                    offline.as_ref(),
                )
                .await
            }
            _ => Err(DecisionError::bad_request(
                "'questions' must be either a JSON array (Flare) or JSON object (Eider)",
            )),
        };
        return match out {
            Ok(json_val) => Json(json_val).into_response(),
            Err(err) => openai_error_response(err.status, err.message),
        };
    }

    // Flat `choices` convenience mode.
    let flat: FlatDecisionFields = match serde_json::from_value(parsed.clone()) {
        Ok(f) => f,
        Err(_) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                "Request must include either 'questions' (array/object) or 'choices' (array)"
                    .to_string(),
            );
        }
    };
    let resolved = match logprob_token_ids(&state.tokenizer, &flat.choices) {
        Ok(r) => r,
        Err(msg) => return openai_error_response(StatusCode::BAD_REQUEST, msg),
    };
    if flat.resolve_only {
        return Json(serde_json::json!({
            "object": "decision.tokens",
            "model": state.model_name,
            "choices": resolved,
        }))
        .into_response();
    }

    let (lp_map, floor) = if let Some(pair) = offline {
        pair
    } else {
        let base_messages = match format_context_text(obj) {
            Ok(Some(msgs)) => msgs,
            Ok(None) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    "Flat choice scoring requires 'messages', 'prompt', 'input', or 'token_logprobs'"
                        .to_string(),
                );
            }
            Err(msg) => return openai_error_response(StatusCode::BAD_REQUEST, msg),
        };
        let items: Vec<OptionItem> = flat
            .choices
            .iter()
            .map(|c| OptionItem {
                label: c.clone(),
                description: None,
            })
            .collect();
        match evaluate_options(
            Some(&state),
            &state.tokenizer,
            &base_messages,
            "",
            &items,
            None,
        )
        .await
        {
            Ok(scored) => {
                let best = flat.choices[scored.best_index].clone();
                let detailed: Vec<Value> = resolved
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        serde_json::json!({
                            "choice": r.choice,
                            "token_ids": r.token_ids,
                            "candidate_first_tokens": r.candidate_first_tokens,
                            "logprob": scored.raw_logprobs[i],
                            "probability": scored.probabilities[i],
                        })
                    })
                    .collect();
                return Json(serde_json::json!({
                    "object": "decision",
                    "model": state.model_name,
                    "choice": best,
                    "selected": best,
                    "selected_index": scored.best_index,
                    "probabilities": scored.probabilities,
                    "confidence": scored.confidence,
                    "concentration": scored.concentration,
                    "choices": detailed,
                }))
                .into_response();
            }
            Err(err) => return openai_error_response(err.status, err.message),
        }
    };

    let groups: Vec<Vec<u32>> = resolved
        .iter()
        .map(|r| r.candidate_first_tokens.clone())
        .collect();
    let scored = score_candidate_groups(&lp_map, floor, &groups);
    let best = flat.choices[scored.best_index].clone();
    let detailed: Vec<Value> = resolved
        .iter()
        .enumerate()
        .map(|(i, r)| {
            serde_json::json!({
                "choice": r.choice,
                "token_ids": r.token_ids,
                "candidate_first_tokens": r.candidate_first_tokens,
                "logprob": scored.raw_logprobs[i],
                "probability": scored.probabilities[i],
            })
        })
        .collect();
    Json(serde_json::json!({
        "object": "decision",
        "model": state.model_name,
        "choice": best,
        "selected": best,
        "selected_index": scored.best_index,
        "probabilities": scored.probabilities,
        "confidence": scored.confidence,
        "concentration": scored.concentration,
        "choices": detailed,
    }))
    .into_response()
}

#[cfg(test)]
#[path = "decisions_tests.rs"]
mod tests;
