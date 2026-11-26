// SPDX-License-Identifier: AGPL-3.0-only

//! OpenAI-compatible SSE streaming chunk types.
//!
//! Split out of `chat_response.rs` (2026-05-19) so each file
//! stays under the 500-LoC cap. Re-exported via `openai::*`;
//! no call site changes (all refs go through the glob).

use serde::Serialize;

use super::*;

// ── SSE streaming types (OpenAI format) ──

/// SSE streaming chunk.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Opt-in report-only serving intervention ledger. Present only on the
    /// terminal finish-bearing chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interventions: Option<InterventionLedger>,
    /// Nonced server-produced prompt evidence, attached only to the terminal
    /// finish-bearing event for explicitly opted-in benchmark requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered_prompt_evidence: Option<RenderedPromptEvidence>,
    /// Server warnings about how the request was served (`max_tokens`
    /// clamped, request timeout). Only on the terminal finish-bearing chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
    pub logprobs: Option<ChoiceLogprobs>,
    /// Exact sampled token IDs (vLLM-compatible `return_token_ids`). Generic
    /// chat places the scheduler-owned sequence on the terminal choice only.
    /// Skipped when empty so opted-out wire bytes stay unchanged.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub token_ids: Vec<u32>,
    /// Non-standard: the server guard behind a `"stop"` (`"watchdog"` |
    /// `"catastrophic_guard"`) or `"length"` (`"request_timeout_stall"` |
    /// `"request_timeout_total"`) finish. Omitted otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flux_stop_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Reasoning trace chunk (from <think>...</think>). Streamed during
    /// the thinking phase when enable_thinking=true. Cline and Roo Code
    /// both check for this field via `"reasoning_content" in delta`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Forward-compatible alias: mirrors `reasoning_content` for clients that
    /// use the shorter `reasoning` field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Omitted when absent — Cline, Roo Code, and most OpenAI-compatible clients
    /// expect content to be missing (not `null`) in role and done chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::tool_parser::ChunkToolCall>>,
    /// Refusal signal. Atlas emits this on a single terminal delta chunk
    /// (just before the `done` chunk) when the accumulated streamed
    /// content matches a known refusal pattern. OpenAI's streaming
    /// refusal model is fragment-by-fragment; Atlas only classifies
    /// post-hoc so the signal lands as one chunk. Safety-aware clients
    /// that branch on `delta.refusal` will still see a non-null value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl ChatCompletionChunk {
    /// First chunk: sends the assistant role.
    pub fn role_chunk(model: &str, id: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".to_string()),
                    reasoning_content: None,
                    reasoning: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                token_ids: Vec::new(),
                flux_stop_reason: None,
            }],
            usage: None,
            interventions: None,
            rendered_prompt_evidence: None,
            warnings: None,
        }
    }

    /// Reasoning content delta chunk (from <think>...</think> when enable_thinking=true).
    /// Cline and Roo Code check for `delta.reasoning_content` in streaming.
    pub fn reasoning_chunk(model: &str, id: &str, text: String) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: Some(text.clone()),
                    reasoning: Some(text),
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                token_ids: Vec::new(),
                flux_stop_reason: None,
            }],
            usage: None,
            interventions: None,
            rendered_prompt_evidence: None,
            warnings: None,
        }
    }

    /// Content delta chunk.
    pub fn content_chunk(model: &str, id: &str, text: String) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    reasoning: None,
                    content: Some(text),
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                token_ids: Vec::new(),
                flux_stop_reason: None,
            }],
            usage: None,
            interventions: None,
            rendered_prompt_evidence: None,
            warnings: None,
        }
    }

    /// Choice-bearing progress frame for a committed token whose decoded
    /// bytes are still buffered or intentionally suppressed. Continuous usage
    /// is attached by the stream tracker before serialization.
    pub fn progress_chunk(model: &str, id: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    reasoning: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                token_ids: Vec::new(),
                flux_stop_reason: None,
            }],
            usage: None,
            interventions: None,
            rendered_prompt_evidence: None,
            warnings: None,
        }
    }

    /// Tool call start chunk — emits role, id, type, name with empty arguments.
    /// Carries `role: "assistant"` alongside the tool metadata; `content` is
    /// omitted from JSON (`None` with `skip_serializing_if = "Option::is_none"`).
    pub fn tool_call_start_chunk(
        model: &str,
        id: &str,
        tc: &crate::tool_parser::ToolCall,
        tc_index: usize,
    ) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".to_string()),
                    reasoning_content: None,
                    reasoning: None,
                    content: None,
                    tool_calls: Some(vec![crate::tool_parser::ChunkToolCall {
                        index: tc_index,
                        id: Some(tc.id.clone()),
                        call_type: Some(tc.call_type.clone()),
                        function: crate::tool_parser::ChunkFunction {
                            name: Some(tc.function.name.clone()),
                            arguments: String::new(),
                        },
                    }]),
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                token_ids: Vec::new(),
                flux_stop_reason: None,
            }],
            usage: None,
            interventions: None,
            rendered_prompt_evidence: None,
            warnings: None,
        }
    }

    /// Tool call argument fragment chunk — emits a partial arguments string.
    /// Per OpenAI streaming spec, subsequent deltas carry incremental argument
    /// fragments. Callers should split the full arguments into small pieces
    /// (~20 chars) and call this for each fragment.
    pub fn tool_call_args_fragment(model: &str, id: &str, tc_index: usize, fragment: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    reasoning: None,
                    content: None,
                    tool_calls: Some(vec![crate::tool_parser::ChunkToolCall {
                        index: tc_index,
                        id: None,
                        call_type: None,
                        function: crate::tool_parser::ChunkFunction {
                            name: None,
                            arguments: fragment.to_string(),
                        },
                    }]),
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                token_ids: Vec::new(),
                flux_stop_reason: None,
            }],
            usage: None,
            interventions: None,
            rendered_prompt_evidence: None,
            warnings: None,
        }
    }

    /// Final chunk with finish_reason, empty delta, and usage.
    pub fn done_chunk(model: &str, id: &str, finish_reason: &str, usage: Usage) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    reasoning: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: Some(finish_reason.to_string()),
                logprobs: None,
                token_ids: Vec::new(),
                flux_stop_reason: None,
            }],
            usage: Some(usage),
            interventions: None,
            rendered_prompt_evidence: None,
            warnings: None,
        }
    }

    /// When `stream_options.include_usage=true`, OpenAI emits a separate
    /// chunk with `choices:[]` and populated `usage` after the final
    /// `finish_reason` choice and before `[DONE]`.
    pub fn usage_only_chunk(model: &str, id: &str, usage: Usage) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: Vec::new(),
            usage: Some(usage),
            interventions: None,
            rendered_prompt_evidence: None,
            warnings: None,
        }
    }

    /// Delta chunk carrying only `refusal: "<sentence>"`. Atlas emits
    /// this once, just before the terminal `done`/`usage_only` chunk,
    /// when `refusal::detect` classifies the accumulated streamed
    /// content as a refusal. OpenAI's streaming refusal model sends
    /// multiple `delta.refusal` fragments; we send a single post-hoc
    /// signal because Atlas classifies after the stream is complete.
    pub fn refusal_chunk(model: &str, id: &str, refusal: String) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    reasoning: None,
                    content: None,
                    tool_calls: None,
                    refusal: Some(refusal),
                },
                finish_reason: None,
                logprobs: None,
                token_ids: Vec::new(),
                flux_stop_reason: None,
            }],
            usage: None,
            interventions: None,
            rendered_prompt_evidence: None,
            warnings: None,
        }
    }

    /// Final chunk carrying only `finish_reason`, with `usage:null`. Used
    /// when `include_usage=true`; the choices-empty usage chunk follows it.
    pub fn final_chunk_no_usage(model: &str, id: &str, finish_reason: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    reasoning: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: Some(finish_reason.to_string()),
                logprobs: None,
                token_ids: Vec::new(),
                flux_stop_reason: None,
            }],
            usage: None,
            interventions: None,
            rendered_prompt_evidence: None,
            warnings: None,
        }
    }

    /// Stamp exact sampled token IDs onto this chunk's first choice. No-op
    /// when empty, so opted-out requests keep a byte-identical wire format.
    pub(crate) fn with_token_ids(mut self, ids: Vec<u32>) -> Self {
        if !ids.is_empty()
            && let Some(choice) = self.choices.first_mut()
        {
            choice.token_ids = ids;
        }
        self
    }

    /// Name the guard behind a `"stop"` finish on the first choice.
    pub(crate) fn with_flux_stop_reason(mut self, reason: Option<&'static str>) -> Self {
        if let Some(choice) = self.choices.first_mut() {
            choice.flux_stop_reason = reason;
        }
        self
    }

    pub(crate) fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    pub(crate) fn with_interventions(mut self, interventions: Option<InterventionLedger>) -> Self {
        self.interventions = interventions;
        self
    }

    pub(crate) fn with_rendered_prompt_evidence(
        mut self,
        evidence: Option<RenderedPromptEvidence>,
    ) -> Self {
        self.rendered_prompt_evidence = evidence;
        self
    }
}
