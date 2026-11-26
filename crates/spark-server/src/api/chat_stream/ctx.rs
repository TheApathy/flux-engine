// SPDX-License-Identifier: AGPL-3.0-only
//
// Read-only context shared by every `StreamEvent` arm. Owned by the
// `flat_map` closure so the per-event handlers can borrow it
// immutably alongside `&mut StreamState`.

use std::sync::Arc;

use crate::AppState;
use crate::tool_parser;

use super::super::failures::F39FailureCache;

pub(super) struct StreamCtx {
    pub(super) state: Arc<AppState>,
    pub(super) model: String,
    pub(super) id: String,
    pub(super) prompt_len: usize,
    pub(super) min_tokens: usize,
    pub(super) ignore_eos: bool,
    pub(super) enable_thinking: bool,
    pub(super) tool_defs_for_backfill: Vec<tool_parser::ToolDefinition>,
    /// Native GLM only: the exact tool set bound into the request's tool
    /// grammar (a specific tool_choice narrows it). `None` for other parsers.
    pub(super) native_tool_defs: Option<Vec<tool_parser::ToolDefinition>>,
    pub(super) cwd_for_normalize: Option<String>,
    pub(super) stop_strings: Vec<String>,
    pub(super) leak_markers: tool_parser::LeakMarkers,
    pub(super) max_tool_calls_per_response: usize,
    /// Whether opt-in agent heuristics (`FLUX_AGENT_GUARDS=1`) such as tool
    /// dedup, F44 permanent-failure blocking, and same-name caps are active.
    pub(super) agent_guards: bool,
    /// OpenAI request contract. `false` ends generation after the first valid,
    /// complete tool call; omitted/`true` preserves multi-call publication.
    pub(super) allow_parallel_tool_calls: bool,
    pub(super) req_stream_include_usage: bool,
    pub(super) req_return_token_ids: bool,
    pub(super) req_ctx: Option<crate::rate_limiter::RequestContext>,
    pub(super) dump_seq: Option<u64>,
    pub(super) f44_cache: F39FailureCache,
    pub(super) f44_cache_active: bool,
    pub(super) intervention_ledger: Option<crate::openai::SharedInterventionLedger>,
    pub(super) rendered_prompt_evidence: Option<crate::openai::RenderedPromptEvidence>,
    /// Request-lifecycle warnings (e.g. `max_tokens` clamped), echoed on
    /// the terminal finish-bearing chunk.
    pub(super) contract_warnings: Vec<String>,
    pub(super) active_guard: std::sync::Arc<std::sync::atomic::AtomicBool>,
}
