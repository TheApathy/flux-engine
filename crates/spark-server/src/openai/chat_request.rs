// SPDX-License-Identifier: AGPL-3.0-only

use serde::Deserialize;

use super::*;

/// Chat completion request (subset of OpenAI spec).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    /// Optional: the DeepSeek-V4.1 Python server ignores `model` and accepts a
    /// request without it; an absent model is the base model (responses echo
    /// the served name either way).
    #[serde(default)]
    pub model: String,
    pub messages: Vec<IncomingMessage>,
    #[serde(
        default = "default_max_tokens",
        deserialize_with = "deserialize_max_tokens_or_default"
    )]
    pub max_tokens: usize,
    /// OpenAI `max_completion_tokens` alias. Accepted alongside `max_tokens`
    /// (and `null` for either); when present, takes precedence over `max_tokens`.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    /// `max_tokens` came from the server's `--default-max-tokens`, not the
    /// client. Such an allowance may be clamped to the remaining context.
    #[serde(skip)]
    pub max_tokens_defaulted: bool,
    /// The server's request defaults for this request, taken ONCE at
    /// admission (`/v1/flux/settings`). Every default this request resolves
    /// (thinking, budgets, sampling presets, `max_tokens`) comes from this one
    /// snapshot, so a settings change never splits a request.
    #[serde(skip)]
    pub runtime_defaults: Option<std::sync::Arc<crate::runtime_settings::RequestDefaults>>,
    pub temperature: Option<f32>,
    /// Top-k: keep only the k highest-probability tokens before sampling.
    /// None = use server default from generation_config.json.
    pub top_k: Option<u32>,
    /// Top-p (nucleus): keep smallest set of tokens whose cumulative probability >= p.
    /// None = use server default from generation_config.json.
    pub top_p: Option<f32>,
    /// Top-n-sigma: filter tokens in logit space before temperature scaling.
    /// Keep only tokens with logit >= mean - n*sigma. Temperature-invariant.
    /// None = use server default. 0.0 = disabled.
    pub top_n_sigma: Option<f32>,
    /// Min-p: keep tokens with prob >= min_p * max_prob (post-softmax).
    /// None = use server default. 0.0 = disabled.
    pub min_p: Option<f32>,
    /// Repetition penalty: penalize tokens that have already been generated.
    /// None = use server default. 1.0 = disabled.
    pub repetition_penalty: Option<f32>,
    /// Presence penalty (OpenAI-style): flat additive penalty for each token that
    /// appeared at least once. Range [-2.0, 2.0], default 0.0 (disabled).
    /// Positive values encourage topic diversity.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty (OpenAI-style): additive penalty proportional to occurrence
    /// count. Range [-2.0, 2.0], default 0.0 (disabled).
    /// Positive values discourage token repetition.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Per-token logit bias: {"token_id": bias_value}. Positive boosts, negative suppresses.
    /// Applied additively to logits before sampling. OpenAI-compatible.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    #[serde(default)]
    pub stream: bool,
    /// Emit the exact sampled token IDs on the terminal streamed choice's
    /// `token_ids` (vLLM-compatible extension). Lets a
    /// benchmark harness count `usage.completion_tokens` precisely
    /// instead of re-tokenizing detokenized text (which over-counts,
    /// since BPE is not homomorphic over fragment concatenation).
    /// PCND: defaults false — opt-in only, so the default wire format
    /// for every existing client stays byte-identical.
    #[serde(default)]
    pub return_token_ids: bool,
    /// Include a report-only intervention ledger on the terminal response.
    /// This never disables or weakens serving guards.
    #[serde(default)]
    pub report_interventions: bool,
    /// Opt-in benchmark challenge for server-produced evidence of the exact
    /// rendered prompt dispatched to the scheduler. This is deliberately
    /// separate from ordinary response metadata because prompts can contain
    /// sensitive data. Accepted only as 32 lowercase hex characters.
    #[serde(default)]
    pub rendered_prompt_evidence_nonce: Option<String>,
    /// Request-local events collected before scheduler dispatch. This is not
    /// part of the input schema and is published only when opted in above.
    #[serde(skip)]
    pub intervention_ledger: InterventionLedger,
    /// Legacy boolean opt-in for chain-of-thought reasoning.
    /// `true`: enables thinking (deferring budget to model policy).
    /// `false` (default): falls through to the model's configured default in
    /// `MODEL.toml` / runtime settings.
    #[serde(default)]
    pub enable_thinking: bool,
    /// Anthropic-style thinking budget: `{"thinking": {"budget_tokens": N}}`
    /// Hard limit on thinking tokens before forcing `</think>`.
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    /// vLLM PR-style thinking budget (top-level integer).
    #[serde(default)]
    pub thinking_token_budget: Option<u32>,
    /// Direct top-level thinking token budget (`{"thinking_budget": N}`).
    #[serde(default)]
    pub thinking_budget: Option<u32>,
    /// Per-request thinking reserve (`{"thinking_reserve": N}`): reserves
    /// `max_tokens` for the visible answer by extending total generation budget
    /// by up to `N` (clamped to remaining context) and capping `thinking_budget <= N`.
    #[serde(default)]
    pub thinking_reserve: Option<u32>,
    /// OpenAI-style reasoning effort: `{"reasoning": {"effort": "low"}}`
    /// A non-object `reasoning` is ignored (as the DeepSeek-V4.1 Python server
    /// does), not a parse error.
    #[serde(default, deserialize_with = "reasoning_object_or_ignore")]
    pub reasoning: Option<ReasoningConfig>,
    /// vLLM-style chat template kwargs: `{"chat_template_kwargs": {"enable_thinking": true}}`
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// Tool definitions for function calling (OpenAI-compatible).
    #[serde(default)]
    pub tools: Option<Vec<crate::tool_parser::ToolDefinition>>,
    /// Tool choice: "auto" (default), "none", "required", or specific function.
    #[serde(default)]
    pub tool_choice: Option<crate::tool_parser::ToolChoice>,
    /// Stop sequences: generation stops when any of these strings is produced.
    /// Accepts a single string or array of strings (OpenAI spec).
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Vec<String>,
    /// Response format constraint (OpenAI-compatible).
    /// `{"type":"text"}` = unconstrained (default),
    /// `{"type":"json_object"}` = any valid JSON,
    /// `{"type":"json_schema","json_schema":{...}}` = JSON matching a schema.
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// Minimum number of tokens to generate before allowing EOS/stop.
    /// 0 = no minimum (default). Useful for preventing empty responses.
    #[serde(default)]
    pub min_tokens: usize,
    /// vLLM/SGLang extension sent by benchmark clients (llm-inference-bench):
    /// model EOS ids and the ChatML role boundary do not end the turn; only
    /// `max_tokens`, `stop` strings and stop tokens do. EOS ids are sampled
    /// and counted like any other token, as in vLLM. Leaves `min_tokens` alone.
    #[serde(default)]
    pub ignore_eos: bool,
    /// Seed for deterministic sampling. When set, stochastic sampling uses this
    /// seed for the RNG, producing reproducible output for the same inputs.
    /// None = non-deterministic (default).
    pub seed: Option<u64>,
    /// Whether to return log-probabilities. OpenAI SDK sends this as a boolean;
    /// Atlas uses `top_logprobs` for the count. Accepted for compatibility but
    /// the actual count is controlled by `top_logprobs`.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of top log-probabilities to return per token (0-20). None = disabled.
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    /// Request timeout in seconds. None = server default.
    #[serde(default)]
    pub timeout: Option<f32>,
    /// Number of chat completion choices to generate (default 1).
    /// Only supported in blocking (non-streaming) mode.
    #[serde(default = "default_n")]
    pub n: usize,
    /// Stream options (OpenAI-compatible). When `include_usage=true`,
    /// a final `choices:[]` chunk with populated `usage` is emitted before
    /// `[DONE]`. `include_obfuscation` defaults true on OpenAI; here we
    /// accept the field but do not emit padding (no side-channel risk on
    /// self-hosted deployments).
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Whether the model may call multiple tools in one turn (OpenAI default
    /// `true`). When explicitly false, Atlas publishes at most the first valid,
    /// complete tool call and cancels the remaining generation.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Controls response length beyond `max_tokens` on gpt-5.x class models
    /// (`low | medium | high`). Atlas accepts the field for compatibility
    /// but does not currently steer output length on top of `max_tokens`.
    #[serde(default)]
    pub verbosity: Option<String>,
    /// Service tier (`auto | default | flex | scale | priority`). Atlas runs
    /// one tier only — accepted for compatibility, echoed back in response.
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Persist the completion for later retrieval via GET `/v1/chat/completions/{id}`.
    /// When `Some(true)`, non-streaming completions are persisted to the in-memory
    /// completion store (`chat_blocking.rs`).
    #[serde(default)]
    pub store: Option<bool>,
    /// User-supplied metadata (≤16 key/value pairs, value ≤512 chars).
    /// Echoed back in the response. OpenAI uses these for completion store
    /// filtering; Atlas just round-trips them.
    #[serde(default)]
    pub metadata: Option<std::collections::HashMap<String, String>>,
    /// Stable identifier for end-users (abuse detection). Atlas accepts,
    /// ignores; kept for back-compat with the deprecated `user` field.
    #[serde(default)]
    pub safety_identifier: Option<String>,
    /// Key used by OpenAI to cache prompt prefixes across requests. Atlas's
    /// prefix cache is content-addressed (hash of prompt tokens), so this
    /// field is accepted and ignored.
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    /// Deprecated (per OpenAI spec) — replaced by `safety_identifier` and
    /// `prompt_cache_key`. Accepted for back-compat with older SDK versions.
    #[serde(default)]
    pub user: Option<String>,
    /// Output modalities requested by the client (`["text"]`, `["text",
    /// "audio"]`, …). Atlas only emits text — when audio is requested
    /// we log a warning and return text only. Accepted for compat with
    /// the gpt-4o-audio / gpt-5-audio family SDKs.
    #[serde(default)]
    pub modalities: Option<Vec<String>>,
    /// Audio-output configuration (voice + format). Atlas does not
    /// serve audio; the field is accepted and ignored so clients that
    /// unconditionally attach it don't 4xx.
    #[serde(default)]
    pub audio: Option<serde_json::Value>,
    /// Predicted Outputs — a hint that large parts of the response are
    /// known ahead of time (e.g. regenerating a file with one edit).
    /// Atlas does not currently run speculative decoding against the
    /// prediction; accepted and ignored. Dropping vs rejecting matches
    /// OpenAI's forward-compat behavior on models that don't support it.
    #[serde(default)]
    pub prediction: Option<serde_json::Value>,
    /// Web-search tool configuration (`web_search_options: {...}`).
    /// Atlas has no web-search backend — accepted and ignored.
    #[serde(default)]
    pub web_search_options: Option<serde_json::Value>,
    /// Reasoning-effort shorthand (`none | minimal | low | medium | high | xhigh`
    /// or integer `1..=100` mapped proportionally to thinking budget).
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// The request body as received, kept only when the served model renders
    /// from the wire format itself (deepseek_v41, `api/dsv41.rs`). Never part
    /// of the wire schema.
    #[serde(skip)]
    pub raw_body: Option<std::sync::Arc<serde_json::Value>>,
}

/// Reasoning effort accepted either as a named OpenAI/Anthropic tier
/// (`"none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"`)
/// or as a DeepSeek-V4.1 integer percentage (`1..=100`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningEffort {
    Named(String),
    Integer(u8),
}

impl ReasoningEffort {
    /// String form passed to Jinja chat templates (`qwen38_reasoning_effort`, etc.).
    #[must_use]
    pub fn as_template_str(&self) -> &str {
        match self {
            Self::Named(s) => s.as_str(),
            Self::Integer(1..=33) => "low",
            Self::Integer(34..=66) => "medium",
            Self::Integer(_) => "xhigh",
        }
    }

    /// Integer percentage (`1..=100`) when supplied as a numeric JSON value.
    #[must_use]
    pub fn as_integer(&self) -> Option<u8> {
        match self {
            Self::Integer(v) => Some(*v),
            Self::Named(_) => None,
        }
    }
}

impl<'de> Deserialize<'de> for ReasoningEffort {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        match serde_json::Value::deserialize(d)? {
            serde_json::Value::String(s) => Ok(Self::Named(s)),
            serde_json::Value::Number(n) if n.is_i64() || n.is_u64() => {
                let Some(v) = n.as_i64().filter(|&v| (1..=100).contains(&v)) else {
                    return Err(D::Error::custom(format!(
                        "integer reasoning effort must be within 1..=100, got {n}"
                    )));
                };
                Ok(Self::Integer(v as u8))
            }
            other => Err(D::Error::custom(format!(
                "reasoning effort must be a string or an integer 1..=100, got {other}"
            ))),
        }
    }
}

/// Map an integer reasoning effort (`1..=100`) proportionally onto `ceiling`.
#[must_use]
pub fn proportional_effort_budget(effort_pct: u8, ceiling: u32) -> u32 {
    if ceiling == 0 || effort_pct == 0 {
        return 0;
    }
    let pct = u64::from(effort_pct.min(100));
    ((u64::from(ceiling) * pct).div_ceil(100)).clamp(1, u64::from(u32::MAX)) as u32
}

fn reasoning_object_or_ignore<'de, D>(d: D) -> Result<Option<ReasoningConfig>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    match Option::<serde_json::Value>::deserialize(d)? {
        Some(v @ serde_json::Value::Object(_)) => serde_json::from_value(v)
            .map(Some)
            .map_err(D::Error::custom),
        _ => Ok(None),
    }
}

/// Stream options (OpenAI-compatible).
#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(default)]
pub struct StreamOptions {
    /// Emit a final chunk with `choices:[]` and populated `usage` before `[DONE]`.
    pub include_usage: bool,
    /// Attach cumulative usage to every streamed choice when `include_usage`
    /// is also true. This vLLM/SGLang extension lets duration-based benchmark
    /// clients measure exact generated-token deltas before completion.
    pub continuous_usage_stats: bool,
    /// Include a random-padding `obfuscation` field on each chunk. Accepted
    /// but not emitted on Atlas — no multi-tenant side-channel risk to defend.
    pub include_obfuscation: bool,
}

/// Response format constraint (OpenAI-compatible).
///
/// Discriminated by `"type"` field:
/// - `"text"`: no constraint (default behavior)
/// - `"json_object"`: output must be valid JSON
/// - `"json_schema"`: output must match the provided JSON schema
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: JsonSchemaSpec },
}

/// JSON schema specification for `response_format.type = "json_schema"`.
#[derive(Debug, Deserialize)]
pub struct JsonSchemaSpec {
    /// Schema name (required by OpenAI spec, used for logging).
    pub name: String,
    /// Optional human-readable description.
    #[serde(default)]
    pub description: Option<String>,
    /// The JSON Schema object.
    pub schema: serde_json::Value,
    /// Whether to enforce strict schema adherence (default: true).
    #[serde(default = "default_true")]
    pub strict: bool,
}

fn default_true() -> bool {
    true
}

/// Anthropic-style thinking configuration.
#[derive(Debug, Deserialize)]
pub struct ThinkingConfig {
    /// Hard token budget for thinking. `Some(0)` enables thinking with a
    /// zero-token budget; use `type: "disabled"` to disable thinking.
    pub budget_tokens: Option<u32>,
    /// "enabled", "disabled", or "adaptive"
    #[serde(rename = "type")]
    pub thinking_type: Option<String>,
}

/// OpenAI-style reasoning configuration.
#[derive(Debug, Deserialize)]
pub struct ReasoningConfig {
    /// Qualitative effort level or DeepSeek-V4.1 integer `1..=100`.
    #[serde(default)]
    pub effort: Option<ReasoningEffort>,
}

/// vLLM-style chat template kwargs.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatTemplateKwargs {
    pub enable_thinking: Option<bool>,
    pub thinking_budget: Option<u32>,
    /// Dense Qwen template control. `None` deliberately leaves the Jinja
    /// variable undefined so the checkpoint's own default applies.
    pub preserve_thinking: Option<bool>,
}

impl ChatTemplateKwargs {
    /// Parse from a JSON string. Returns `None` if parsing fails or string is empty.
    pub fn from_json(s: &str) -> Option<Self> {
        if s.trim().is_empty() {
            return None;
        }
        serde_json::from_str(s).ok()
    }
}

/// Preserve the existing budget for unrecognised reasoning-effort values.
const UNKNOWN_EFFORT_THINKING_BUDGET: u32 = 256;

/// Thinking budgets for templates that render the Qwen3.8 effort vocabulary
/// (`xhigh | medium | low`, see `tokenizer::chat_impl::qwen38_reasoning_effort`).
/// Each budget follows the instruction the template renders for that effort:
/// `low` asks for brief thinking, `medium` renders no instruction (the model's
/// own length), and `high`/`xhigh`/`max` render the same xhigh "think
/// carefully" instruction as an omitted effort, so they get the same
/// model-policy budget (`None`). Unknown values render the template's xhigh
/// default and are budgeted the same way.
const QWEN38_LOW_EFFORT_THINKING_BUDGET: u32 = 1024;
const QWEN38_MEDIUM_EFFORT_THINKING_BUDGET: u32 = 8192;

pub fn qwen38_effort_thinking_budget(effort: &str) -> Option<u32> {
    match effort {
        "none" => Some(0),
        "minimal" | "low" => Some(QWEN38_LOW_EFFORT_THINKING_BUDGET),
        "medium" => Some(QWEN38_MEDIUM_EFFORT_THINKING_BUDGET),
        _ => None,
    }
}

impl ChatCompletionRequest {
    pub(crate) fn validate_rendered_prompt_evidence_nonce(&self) -> Result<(), &'static str> {
        let Some(nonce) = self.rendered_prompt_evidence_nonce.as_deref() else {
            return Ok(());
        };
        if !self.report_interventions {
            return Err("rendered_prompt_evidence_nonce requires report_interventions=true");
        }
        if nonce.len() != 32
            || !nonce
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(
                "rendered_prompt_evidence_nonce must be exactly 32 lowercase hex characters",
            );
        }
        Ok(())
    }

    pub(crate) fn record_intervention(&mut self, code: &str, action: &str) {
        if self.report_interventions {
            self.intervention_ledger.record(code, action);
        }
    }

    /// Accept both OpenAI reasoning-effort request spellings. The nested
    /// object wins when both are present.
    pub fn requested_reasoning_effort_value(&self) -> Option<&ReasoningEffort> {
        self.reasoning
            .as_ref()
            .and_then(|r| r.effort.as_ref())
            .or(self.reasoning_effort.as_ref())
    }

    /// Template-visible effort string (`"low"`, `"medium"`, `"high"`, `"xhigh"`, etc.).
    pub fn requested_reasoning_effort(&self) -> Option<&str> {
        self.requested_reasoning_effort_value()
            .map(ReasoningEffort::as_template_str)
    }

    /// Integer reasoning effort (`1..=100`) if the client supplied an integer
    /// and no higher-priority explicit thinking budget channel overrides it.
    pub fn requested_reasoning_effort_int(&self) -> Option<u8> {
        if self.thinking.is_some()
            || self.thinking_token_budget.is_some()
            || self.thinking_budget.is_some()
        {
            return None;
        }
        self.requested_reasoning_effort_value()
            .and_then(ReasoningEffort::as_integer)
    }

    /// Resolve thinking parameters from all supported request-body formats
    /// into a single `(enable_thinking: bool, thinking_budget: Option<u32>)`
    /// pair. The client's per-request choice always wins over the model
    /// default; the model default (from MODEL.toml `[behavior].thinking_default`)
    /// is used only when the client sends NO thinking parameter at all.
    ///
    /// The `--disable-thinking` CLI flag is a higher-priority kill switch
    /// applied by the caller (api.rs / anthropic.rs) — this function does
    /// not know about it.
    ///
    /// Request-body priority (highest to lowest):
    /// 1. `thinking.budget_tokens` (Anthropic) — explicit budget
    /// 2. `thinking_token_budget` (vLLM PR) — explicit budget
    /// 3. `reasoning.effort` (OpenAI) — mapped to budget
    /// 4. `chat_template_kwargs` (vLLM stable) — enable/disable + optional budget
    /// 5. `enable_thinking` (Atlas legacy) — boolean, no explicit budget
    /// 6. `model_default` argument (from MODEL.toml) — model-specific fallback
    ///
    /// Enabling thinking without an explicit budget leaves it unspecified so
    /// the policy layer applies the model profile and generation safety cap.
    /// Unknown effort values retain their existing 256-token fallback.
    ///
    /// Returns `true` if any of channels 1-5 carried an explicit thinking
    /// intent from the client (i.e. the resolved value did NOT fall through
    /// to `model_default`). Callers use this to decide whether a
    /// server-side policy (e.g. `thinking_in_tools=false`) is allowed to
    /// override the model default OR must respect the explicit request.
    pub fn thinking_explicitly_requested(&self) -> bool {
        if self.thinking.is_some() {
            return true;
        }
        if self.thinking_token_budget.is_some() || self.thinking_budget.is_some() {
            return true;
        }
        if let Some(ref rc) = self.reasoning
            && rc.effort.is_some()
        {
            return true;
        }
        if self.reasoning_effort.is_some() {
            return true;
        }
        if let Some(ref kw) = self.chat_template_kwargs
            && (kw.thinking_budget.is_some() || kw.enable_thinking.is_some())
        {
            return true;
        }
        if self.enable_thinking {
            return true;
        }
        false
    }

    pub fn resolve_thinking(&self, model_default: bool) -> (bool, Option<u32>) {
        self.resolve_thinking_with_effort(model_default, &generic_effort_thinking_budget)
    }

    /// [`Self::resolve_thinking`] with a model-specific effort table. The
    /// table returns `None` to defer an effort to the model policy budget.
    pub fn resolve_thinking_with_effort(
        &self,
        model_default: bool,
        effort_budget: &dyn Fn(&str) -> Option<u32>,
    ) -> (bool, Option<u32>) {
        // 1. Anthropic: thinking.budget_tokens
        if let Some(ref tc) = self.thinking {
            if let Some(ref t) = tc.thinking_type
                && t == "disabled"
            {
                return (false, Some(0));
            }
            if let Some(budget) = tc.budget_tokens {
                return (true, Some(budget));
            }
            // Thinking object present with no budget: defer to model policy.
            return (true, None);
        }

        // 2. vLLM PR: thinking_token_budget / top-level thinking_budget
        if let Some(budget) = self.thinking_token_budget.or(self.thinking_budget) {
            return (budget > 0, Some(budget));
        }

        // 3. OpenAI: reasoning.effort
        if let Some(pct) = self.requested_reasoning_effort_int() {
            let budget = proportional_effort_budget(pct, 1024);
            return (budget > 0, Some(budget));
        }
        if let Some(effort) = self.requested_reasoning_effort() {
            return match effort_budget(effort) {
                Some(budget) => (budget > 0, Some(budget)),
                None => (true, None),
            };
        }

        // 4. vLLM stable: chat_template_kwargs
        if let Some(ref kwargs) = self.chat_template_kwargs {
            if let Some(budget) = kwargs.thinking_budget {
                return (budget > 0, Some(budget));
            }
            if let Some(enabled) = kwargs.enable_thinking {
                return if enabled {
                    (true, None)
                } else {
                    (false, Some(0))
                };
            }
        }

        // 5. Atlas legacy: enable_thinking boolean in the request body.
        // Only honored when explicitly true — a false value (including the
        // serde default when the field is absent) falls through to the
        // MODEL.toml default so clients that don't know about this flag
        // inherit the model's design intent instead of silently opting out.
        if self.enable_thinking {
            return (true, None);
        }

        // 6. Model default from MODEL.toml [behavior].thinking_default.
        // Leave the budget unspecified here so the API policy layer uses
        // this model target's `max_thinking_budget`. Returning the generic
        // 256-token request default made per-model budgets (including
        // Qwen3.8's 2048-token profile) silently inert whenever the client
        // omitted an explicit reasoning field.
        if model_default {
            (true, None)
        } else {
            (false, None)
        }
    }
}

/// The model-agnostic effort table.
pub fn generic_effort_thinking_budget(effort: &str) -> Option<u32> {
    Some(match effort {
        "none" => 0,
        "minimal" => 64,
        "low" => 128,
        "medium" => 256,
        "high" => 512,
        "xhigh" | "max" => 1024,
        _ => UNKNOWN_EFFORT_THINKING_BUDGET,
    })
}

/// Whether a request body names its output allowance (`max_tokens` or
/// `max_completion_tokens`). An invalid body reports `true`, so the normal
/// parse error, not a default, decides it.
pub fn body_sets_max_tokens(body: &[u8]) -> bool {
    #[derive(Deserialize)]
    struct Probe {
        max_tokens: Option<serde::de::IgnoredAny>,
        max_completion_tokens: Option<serde::de::IgnoredAny>,
    }
    serde_json::from_slice::<Probe>(body)
        .map(|p| p.max_tokens.is_some() || p.max_completion_tokens.is_some())
        .unwrap_or(true)
}

impl ChatCompletionRequest {
    /// Resolve `max_completion_tokens` precedence and apply the server default
    /// when the client sent no output allowance (or sent `null`).
    pub fn apply_default_max_tokens(&mut self, body_sets_max_tokens: bool, default: Option<usize>) {
        if let Some(mct) = self.max_completion_tokens {
            self.max_tokens = mct;
        }
        if let Some(tokens) = default
            && !body_sets_max_tokens
        {
            self.max_tokens = tokens;
            self.max_tokens_defaulted = true;
        }
    }
}

pub(super) fn default_max_tokens() -> usize {
    4096
}

fn deserialize_max_tokens_or_default<'de, D>(d: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<usize>::deserialize(d)?.unwrap_or_else(default_max_tokens))
}
pub(super) fn default_n() -> usize {
    1
}

/// Deserialize `stop` as null, a single string, or array of strings (OpenAI spec).
pub(super) fn deserialize_stop<'de, D>(d: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum RawStop {
        Str(String),
        Arr(Vec<String>),
        Null(()),
    }
    match RawStop::deserialize(d)? {
        RawStop::Str(s) => Ok(vec![s]),
        RawStop::Arr(v) => Ok(v),
        RawStop::Null(()) => Ok(Vec::new()),
    }
}
