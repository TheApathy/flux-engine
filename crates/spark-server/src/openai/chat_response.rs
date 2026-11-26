// SPDX-License-Identifier: AGPL-3.0-only

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use super::*;

/// Chat completion response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    /// Echo of the request's `service_tier` (OpenAI-compatible).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Echo of the request's `metadata` (OpenAI-compatible).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,
    /// Opt-in, report-only record of serving interventions that changed this
    /// request. Omitted unless the request set `report_interventions=true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interventions: Option<InterventionLedger>,
    /// Server-produced evidence for the exact prompt tokens dispatched for an
    /// explicitly nonced benchmark request. Omitted on ordinary requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered_prompt_evidence: Option<RenderedPromptEvidence>,
    /// Server warnings about how the request was served (for example
    /// `max_tokens` clamped to the context left). Omitted when empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RenderedPromptEvidence {
    pub schema: &'static str,
    pub nonce: String,
    pub response_id: String,
    pub prompt_sha256: String,
}

impl RenderedPromptEvidence {
    pub(crate) fn digest(prompt_utf8: &str) -> String {
        let digest = ring::digest::digest(&ring::digest::SHA256, prompt_utf8.as_bytes());
        digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    pub(crate) fn from_digest(nonce: String, response_id: String, prompt_sha256: String) -> Self {
        Self {
            // Preserved wire-schema identifier for benchmark/repro-kit compatibility.
            schema: "atlas.rendered_prompt.v1",
            nonce,
            response_id,
            prompt_sha256,
        }
    }
}

/// Report-only audit trail for benchmark and quality attribution.
///
/// This deliberately does not provide a switch for disabling guards. A caller
/// can ask which interventions fired without changing the serving policy being
/// measured.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct InterventionLedger {
    pub schema: &'static str,
    pub policy: &'static str,
    pub events: Vec<InterventionEvent>,
}

impl Default for InterventionLedger {
    fn default() -> Self {
        Self {
            // Preserved wire-schema identifier for benchmark/repro-kit compatibility.
            schema: "atlas.interventions.v1",
            policy: "report_only",
            events: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct InterventionEvent {
    pub code: String,
    pub action: String,
    pub count: u32,
}

impl InterventionLedger {
    /// Record one intervention, coalescing repeated identical actions so long
    /// responses do not grow the response envelope without bound.
    pub fn record(&mut self, code: &str, action: &str) {
        if let Some(event) = self
            .events
            .iter_mut()
            .find(|event| event.code == code && event.action == action)
        {
            event.count = event.count.saturating_add(1);
            return;
        }
        self.events.push(InterventionEvent {
            code: code.to_string(),
            action: action.to_string(),
            count: 1,
        });
    }
}

/// Per-request audit state shared by the API and scheduler. It is allocated
/// only for explicitly opted-in requests, leaving the normal hot path alone.
pub type SharedInterventionLedger = Arc<Mutex<InterventionLedger>>;

pub fn record_shared_intervention(
    ledger: Option<&SharedInterventionLedger>,
    code: &str,
    action: &str,
) {
    if let Some(ledger) = ledger {
        ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .record(code, action);
    }
}

pub fn snapshot_shared_interventions(
    ledger: Option<&SharedInterventionLedger>,
) -> Option<InterventionLedger> {
    ledger.map(|ledger| {
        ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    })
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
    pub logprobs: Option<ChoiceLogprobs>,
    /// Scheduler-owned generated token sequence. Opt-in only; omitted by
    /// default so the ordinary OpenAI wire shape remains unchanged.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub token_ids: Vec<u32>,
    /// Non-standard: the server guard behind a `"stop"` (`"watchdog"` |
    /// `"catastrophic_guard"`) or `"length"` (`"request_timeout_stall"` |
    /// `"request_timeout_total"`) finish. Omitted otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flux_stop_reason: Option<&'static str>,
}

/// Token usage and performance timing.
///
/// Standard OpenAI fields (`prompt_tokens`, `completion_tokens`, `total_tokens`)
/// plus timing extensions that OpenWebUI and other frontends display in tooltips.
/// Field naming follows llama.cpp / Ollama conventions for broad compatibility.
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// Prefix-cache + audio token breakdown of the prompt (OpenAI-compatible).
    /// Populated when Atlas's prefix cache served any portion of the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Reasoning + audio + prediction breakdown of the completion
    /// (OpenAI-compatible). `reasoning_tokens` counts the tokens emitted
    /// inside `<think>...</think>` (or the equivalent for each model type).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// Time to first token in milliseconds (prefill duration).
    #[serde(rename = "time_to_first_token_ms")]
    pub time_to_first_token_ms: f64,
    /// Decode throughput in tokens per second.
    #[serde(rename = "response_token/s")]
    pub response_tokens_per_second: f64,
}

/// Prompt-token breakdown (OpenAI-compatible `prompt_tokens_details`).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromptTokensDetails {
    /// Tokens served by the prefix cache (no prefill compute cost).
    pub cached_tokens: usize,
    /// Audio-input tokens. Always 0 on Atlas until audio modality lands.
    pub audio_tokens: usize,
}

/// Completion-token breakdown (OpenAI-compatible `completion_tokens_details`).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompletionTokensDetails {
    /// Tokens generated inside a thinking/reasoning block
    /// (`<think>...</think>`, `[THINK]...[/THINK]`, etc.). Counted in
    /// `completion_tokens` as well — this is the portion attributable to
    /// chain-of-thought.
    pub reasoning_tokens: usize,
    /// Audio-output tokens. Always 0 on Atlas until audio modality lands.
    pub audio_tokens: usize,
    /// Predicted-output (`prediction`) tokens that matched generation.
    /// Always 0 on Atlas — we don't implement predicted outputs yet.
    pub accepted_prediction_tokens: usize,
    /// Predicted-output tokens that were rejected. Always 0 on Atlas.
    pub rejected_prediction_tokens: usize,
    /// Thinking nudges the server injected into the reasoning
    /// (`FLUX_THINK_NUDGE=on`). Omitted when none fired, so responses are
    /// unchanged while the nudge is off.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub think_nudges: u32,
}

fn is_zero_u32(value: &u32) -> bool {
    *value == 0
}

/// Top log-probability for a single alternative token.
#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

/// Log-probability information for a single generated token.
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprobInfo {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprob>,
}

/// Per-choice logprobs container (OpenAI-compatible).
#[derive(Debug, Clone, Serialize)]
pub struct ChoiceLogprobs {
    pub content: Vec<TokenLogprobInfo>,
}

/// Model list response.
#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    /// The served context cap (`--max-seq-len`, the bound completions admit
    /// against), under vLLM's key so clients size prompts from it.
    pub max_model_len: usize,
}

impl ChatCompletionResponse {
    pub fn new(model: &str, content: String, usage: Usage, finish_reason: &str) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid_v4()),
            object: "chat.completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    reasoning_content: None,
                    reasoning: None,
                    annotations: extract_url_annotations(&content),
                    refusal: None,
                    content: Some(content),
                    tool_calls: None,
                },
                finish_reason: finish_reason.to_string(),
                logprobs: None,
                token_ids: Vec::new(),
                flux_stop_reason: None,
            }],
            usage,
            service_tier: None,
            metadata: None,
            interventions: None,
            rendered_prompt_evidence: None,
            warnings: None,
        }
    }

    pub fn with_tool_calls(
        model: &str,
        content: Option<String>,
        tool_calls: Vec<crate::tool_parser::ToolCall>,
        usage: Usage,
    ) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid_v4()),
            object: "chat.completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    reasoning_content: None,
                    reasoning: None,
                    annotations: content.as_deref().and_then(extract_url_annotations),
                    refusal: None,
                    content,
                    tool_calls: Some(tool_calls),
                },
                finish_reason: "tool_calls".to_string(),
                logprobs: None,
                token_ids: Vec::new(),
                flux_stop_reason: None,
            }],
            usage,
            service_tier: None,
            metadata: None,
            interventions: None,
            rendered_prompt_evidence: None,
            warnings: None,
        }
    }
}

#[cfg(test)]
mod intervention_tests {
    use super::*;

    #[test]
    fn ledger_is_report_only_and_coalesces_repeated_actions() {
        let mut ledger = InterventionLedger::default();
        ledger.record("F28", "disable_thinking_after_tool_error");
        ledger.record("F28", "disable_thinking_after_tool_error");

        let value = serde_json::to_value(&ledger).unwrap();
        assert_eq!(value["schema"], "atlas.interventions.v1");
        assert_eq!(value["policy"], "report_only");
        assert_eq!(value["events"][0]["code"], "F28");
        assert_eq!(value["events"][0]["count"], 2);
    }

    #[test]
    fn response_omits_ledger_unless_caller_opted_in() {
        let response = ChatCompletionResponse::new(
            "model",
            "ok".into(),
            Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
                prompt_tokens_details: None,
                completion_tokens_details: None,
                time_to_first_token_ms: 1.0,
                response_tokens_per_second: 1.0,
            },
            "stop",
        );
        assert!(
            !serde_json::to_string(&response)
                .unwrap()
                .contains("interventions")
        );
    }
}
