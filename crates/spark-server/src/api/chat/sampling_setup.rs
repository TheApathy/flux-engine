// SPDX-License-Identifier: AGPL-3.0-only
//
// Sampling-preset selection, stop-token tokenisation, grammar-spec
// construction, and timeout / logprobs resolution.
//
// Lifted out of `chat::chat_completions_inner` (wave 4g).

use axum::http::StatusCode;
use axum::response::Response;
use std::sync::Arc;

use crate::AppState;
use crate::openai::ChatCompletionRequest;
use crate::tool_parser;

use super::super::compact::{openai_error_response, openai_error_response_with_param};
use super::super::inference_impl::tokenize_stop_sequences;
use super::super::inference_types::{GrammarSpec, ResponseFormatConstraint};

pub(super) struct SamplingSetup {
    pub(super) temperature: f32,
    pub(super) top_k: u32,
    pub(super) top_p: f32,
    pub(super) top_n_sigma: f32,
    pub(super) min_p: f32,
    pub(super) repetition_penalty: f32,
    pub(super) presence_penalty: f32,
    pub(super) frequency_penalty: f32,
    pub(super) dry_multiplier: f32,
    pub(super) dry_base: f32,
    pub(super) dry_allowed_length: u32,
    pub(super) lz_penalty: f32,
    pub(super) logit_bias: Vec<(u32, f32)>,
    pub(super) max_tokens: usize,
    pub(super) stop_tokens: Vec<u32>,
    pub(super) tool_choice_required: bool,
    pub(super) grammar_spec: Option<GrammarSpec>,
    pub(super) timeout_at: Option<std::time::Instant>,
    pub(super) top_logprobs: Option<u8>,
}

/// Tool turns use the `tools` preset unless thinking is on and the model opts
/// its reasoning into the thinking preset (`thinking_sampling_in_tools`).
fn select_preset(
    presets: &atlas_kernels::SamplingPresets,
    tools_active: bool,
    enable_thinking: bool,
    thinking_sampling_in_tools: bool,
) -> &atlas_kernels::SamplingCategory {
    if tools_active && !(enable_thinking && thinking_sampling_in_tools) {
        &presets.tools
    } else if enable_thinking {
        &presets.thinking_text
    } else {
        &presets.non_thinking
    }
}

/// Token-history penalties after preset and request resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
struct HistoryPenalties {
    presence: f32,
    frequency: f32,
    dry_multiplier: f32,
    lz: f32,
}

/// A client that pins `temperature: 0.0` asks for greedy decoding, so the
/// preset's history penalties are neutral (they would move the argmax away
/// from the model's top-1). Penalties the client sends itself still apply;
/// DRY and LZ have no request field and come only from the preset.
fn resolve_history_penalties(
    req_temperature: Option<f32>,
    req_presence: Option<f32>,
    req_frequency: Option<f32>,
    preset: &atlas_kernels::SamplingCategory,
) -> HistoryPenalties {
    let greedy = req_temperature == Some(0.0);
    let from_preset = |value: f32| if greedy { 0.0 } else { value };
    HistoryPenalties {
        presence: req_presence.unwrap_or(from_preset(preset.presence_penalty)),
        frequency: req_frequency.unwrap_or(from_preset(preset.frequency_penalty)),
        dry_multiplier: from_preset(preset.dry_multiplier),
        lz: from_preset(preset.lz_penalty),
    }
}

fn resolve_optional_filter(
    request_value: Option<f32>,
    preset_value: Option<f32>,
    server_fallback: f32,
) -> f32 {
    request_value.or(preset_value).unwrap_or(server_fallback)
}

/// Required/specific choices and explicit single-call requests share a
/// terminal grammar. Both blocking and streaming use this sampling setup.
fn stop_after_first_for_request(use_triggers: bool, parallel_tool_calls: Option<bool>) -> bool {
    !use_triggers || parallel_tool_calls == Some(false)
}

const fn tool_call_repeat_logit_bias(tool_call_repeat_count: usize) -> f32 {
    match tool_call_repeat_count {
        0..=2 => 0.0,
        3 => -5.0,
        _ => -10.0,
    }
}

fn checked_request_deadline(
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

/// Post-first-token deadline under the request-lifecycle contract, from the
/// final (possibly clamped) `max_tokens`. `Some` only when the derived total
/// exceeds the server base; an explicit request `timeout`, a base of 0 and
/// the deepseek_v41 no-deadline rule all return `None`, which keeps the
/// pre-first-token deadline for the whole request (byte-identical default).
pub(super) fn contract_decode_deadline(
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

#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)]
pub(super) fn build_sampling(
    state: &Arc<AppState>,
    defaults: &crate::runtime_settings::RequestDefaults,
    req: &ChatCompletionRequest,
    enable_thinking: bool,
    tools_active: bool,
    suppress_tool_call: bool,
    tool_call_repeat_count: usize,
) -> Result<SamplingSetup, Response> {
    // Preset selection.
    // Presets from the request's admission snapshot (`/v1/flux/settings`).
    let preset = select_preset(
        &defaults.sampling_presets,
        tools_active,
        enable_thinking,
        state.behavior.thinking_sampling_in_tools,
    );
    // FLUX_IGNORE_REQUEST_SAMPLING=1 (legacy ATLAS_IGNORE_REQUEST_SAMPLING=1):
    // serve at the model card's presets even when the client pins sampling
    // (e.g. a bench harness sending temperature 0.0). Greedy decoding of long
    // thinking spans is a documented failure mode for this model family
    // (repetition loops, instant think-close); the MODEL.toml presets are the
    // vendor-recommended regime.
    let ignore_req_sampling = {
        static IG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *IG.get_or_init(|| {
            std::env::var("FLUX_IGNORE_REQUEST_SAMPLING")
                .ok()
                .or_else(|| std::env::var("ATLAS_IGNORE_REQUEST_SAMPLING").ok())
                .as_deref()
                == Some("1")
        })
    };
    let (req_temperature, req_top_k, req_top_p) = if ignore_req_sampling {
        (None, None, None)
    } else {
        (req.temperature, req.top_k, req.top_p)
    };
    // FLUX_TEMP_OVERRIDE (legacy ATLAS_TEMP_OVERRIDE): force this temperature
    // (after preset/request resolution). Pairs with FLUX_IGNORE_REQUEST_SAMPLING
    // to serve a fixed sampling regime regardless of what the client pins.
    let temp_override = {
        static T: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
        *T.get_or_init(|| {
            std::env::var("FLUX_TEMP_OVERRIDE")
                .ok()
                .or_else(|| std::env::var("ATLAS_TEMP_OVERRIDE").ok())
                .and_then(|v| v.parse().ok())
        })
    };
    let effective_req_temperature = temp_override.or(req_temperature);
    let temperature = effective_req_temperature.unwrap_or(preset.temperature);
    // deepseek_v41: the Python server samples with temperature, top_p and the
    // presence/frequency penalties only; it ignores top_k, top_n_sigma, min_p
    // and repetition_penalty, so they are neutral here whatever the client sends.
    let dsv41 = state.dsv41;
    let top_k = if dsv41 {
        0
    } else {
        req_top_k.unwrap_or(preset.top_k)
    };
    let top_p = req_top_p.unwrap_or(preset.top_p);
    let top_n_sigma = if dsv41 {
        0.0
    } else {
        resolve_optional_filter(
            req.top_n_sigma,
            preset.top_n_sigma,
            state.default_top_n_sigma,
        )
    };
    let min_p = if dsv41 {
        0.0
    } else {
        resolve_optional_filter(req.min_p, preset.min_p, state.default_min_p)
    };
    let repetition_penalty = if dsv41 {
        1.0
    } else if effective_req_temperature == Some(0.0) {
        req.repetition_penalty.unwrap_or(1.0)
    } else {
        req.repetition_penalty.unwrap_or(preset.repetition_penalty)
    };
    let HistoryPenalties {
        presence: presence_penalty,
        frequency: frequency_penalty,
        dry_multiplier,
        lz: lz_penalty,
    } = resolve_history_penalties(
        effective_req_temperature,
        req.presence_penalty,
        req.frequency_penalty,
        preset,
    );
    let dry_base = preset.dry_base;
    let dry_allowed_length = preset.dry_allowed_length;

    // OpenAI-style penalty range validation.
    if !(-2.0..=2.0).contains(&presence_penalty) {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("presence_penalty must be between -2.0 and 2.0, got {presence_penalty}"),
        ));
    }
    if !(-2.0..=2.0).contains(&frequency_penalty) {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("frequency_penalty must be between -2.0 and 2.0, got {frequency_penalty}"),
        ));
    }

    // Logit bias from OpenAI (string keys) → Vec<(u32, f32)>.
    let mut logit_bias: Vec<(u32, f32)> = req.logit_bias.as_ref().map_or(Vec::new(), |map| {
        map.iter()
            .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
            .collect()
    });

    // Anti-loop `<tool_call>` decay: keep initial `tool_choice="auto"` turns
    // unbiased (0.0, matching vLLM/SGLang calibration so conversational turns
    // do not suffer false-positive or redundant tool calls), and decay only
    // once identical tool calls repeat 3+ times.
    if tools_active
        && !suppress_tool_call
        && super::super::chat_phases::agent_guards_enabled()
        && let Some(tc_id) = state.tool_call_start_token_id
    {
        let bias = tool_call_repeat_logit_bias(tool_call_repeat_count);
        if bias != 0.0 {
            logit_bias.push((tc_id, bias));
        }
    }

    // Same effective ceiling consumed by thinking resolution. Keep the cap in
    // one helper so tool turns cannot budget reasoning against the raw client
    // max and only later shrink the actual generation allowance.
    // deepseek_v41: the Python server has no tool-turn cap.
    let max_tokens = if dsv41 {
        req.max_tokens
    } else {
        super::thinking::generation_max_tokens(req.max_tokens, tools_active, state.tool_max_tokens)
    };
    if tools_active && max_tokens < req.max_tokens {
        tracing::info!(
            "Tool max_tokens cap: {} → {} (tool_max_tokens={})",
            req.max_tokens,
            max_tokens,
            state.tool_max_tokens
        );
    }

    // Stop tokens.
    // Do not inject `</tool_call>` into `stop_tokens` (which merges into
    // `eos_tokens` regardless of thinking state or model format): structural
    // tool-call termination is handled by `tool_call_end_token` via
    // `commit_tool_call_close` only outside `<think>`.
    let stop_tokens = tokenize_stop_sequences(&state.tokenizer, &req.stop);

    // Tool-choice required mode: honor the client's `tool_choice` contract
    // across all parsers (do not override `tool_choice="auto"` to required).
    let tool_choice_required = tools_active
        && req.tool_choice.as_ref().is_some_and(|tc| {
            matches!(tc, tool_parser::ToolChoice::Mode(m) if m == "required")
                || matches!(tc, tool_parser::ToolChoice::Specific { .. })
        });

    // response_format + tools coexistence.
    //
    // OpenAI's API allows both fields in the same request; agentic pipelines
    // routinely set both (the model emits a tool call on turn N, then a
    // schema-shaped final answer on turn N+1):
    //   * `tool_choice="none"` (or no tools) → enforce response_format only.
    //   * `tool_choice="required"` / specific function → enforce tool-call
    //     grammar only (this turn must call a tool, not return final JSON).
    //   * `tool_choice="auto"` (or omitted) with both `tools` and
    //     `response_format` → enforce a joint XGrammar union
    //     `(tool_call(s)) | (response_format JSON)` so turn N can emit a valid
    //     tool call and turn N+1 is constrained to schema-valid JSON (allowing
    //     optional ```json fences, which `repair_json` strips) with zero
    //     surrounding prose.
    let response_format_constraint: Option<ResponseFormatConstraint> =
        match req.response_format.as_ref() {
            Some(crate::openai::ResponseFormat::JsonObject) => {
                Some(ResponseFormatConstraint::JsonObject)
            }
            Some(crate::openai::ResponseFormat::JsonSchema { json_schema }) => {
                if json_schema.strict
                    && let Err(err) =
                        crate::api::schema_check::validate_strict_json_schema(&json_schema.schema)
                {
                    return Err(openai_error_response_with_param(
                        StatusCode::BAD_REQUEST,
                        err.to_string(),
                        Some("response_format.json_schema.schema"),
                        Some("invalid_json_schema"),
                    ));
                }
                Some(ResponseFormatConstraint::JsonSchema {
                    schema: json_schema.schema.to_string(),
                    strict: json_schema.strict,
                })
            }
            Some(crate::openai::ResponseFormat::Text) | None => None,
        };
    let has_response_format = response_format_constraint.is_some();
    let tool_choice_none = req
        .tool_choice
        .as_ref()
        .is_some_and(|tc| matches!(tc, tool_parser::ToolChoice::Mode(m) if m == "none"));
    let response_format_only = has_response_format && (!tools_active || tool_choice_none);

    // Grammar spec (XGrammar structural-tag / JSON-schema / joint union enforcement).
    let use_triggers = !tool_choice_required;
    let grammar_spec: Option<GrammarSpec> = if response_format_only {
        match response_format_constraint {
            Some(ResponseFormatConstraint::JsonObject) => Some(GrammarSpec::JsonObject),
            Some(ResponseFormatConstraint::JsonSchema { schema, strict }) => {
                Some(GrammarSpec::JsonSchema { schema, strict })
            }
            None => None,
        }
    } else if tools_active && state.behavior.disable_tool_grammar {
        // Structure-snowballing escape hatch (arXiv:2604.06066): this
        // model tool-calls more reliably unconstrained. Tool calls are
        // still parsed from the output — just not grammar-enforced.
        tracing::info!("MODEL.toml [behavior].disable_tool_grammar=true — tool-call grammar OFF");
        None
    } else if tools_active {
        let joint_rf = if use_triggers {
            response_format_constraint
        } else {
            None
        };
        if joint_rf.is_some() {
            tracing::info!(
                "response_format + tools both set (tool_choice=auto); enforcing joint \
                 tool-call | response_format union grammar"
            );
        } else if has_response_format {
            tracing::info!(
                "response_format + tools both set (tool_choice=required/specific); \
                 enforcing tool-call grammar for this turn"
            );
        }
        let parser = state.tool_call_parser.as_ref().map(std::sync::Arc::clone);
        let mut tools = req.tools.as_ref().cloned().unwrap_or_default();
        if let Some(tool_parser::ToolChoice::Specific { ref function }) = req.tool_choice {
            tools.retain(|t| t.function.name == function.name);
        }
        parser.map(|p| GrammarSpec::ToolCall {
            tools,
            parser: p,
            use_triggers,
            stop_after_first: stop_after_first_for_request(use_triggers, req.parallel_tool_calls),
            response_format: joint_rf,
        })
    } else {
        None
    };

    // Timeout deadline.
    // deepseek_v41: the Python server has no server-side deadline; only an
    // explicit request `timeout` sets one.
    let timeout_secs = req.timeout.unwrap_or(if dsv41 {
        0.0
    } else {
        state.request_timeout as f32
    });
    let timeout_at =
        checked_request_deadline(timeout_secs, std::time::Instant::now()).map_err(|message| {
            openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                message.to_owned(),
                Some("timeout"),
                Some("invalid_timeout"),
            )
        })?;

    // top_logprobs (OpenAI spec: 0-20).
    let top_logprobs = req.top_logprobs.map(|n| n.min(20));

    Ok(SamplingSetup {
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
        max_tokens,
        stop_tokens,
        tool_choice_required,
        grammar_spec,
        timeout_at,
        top_logprobs,
    })
}

#[cfg(test)]
mod stop_after_first_tests {
    use super::stop_after_first_for_request;

    #[test]
    fn parallel_false_stops_the_shared_auto_grammar_after_one_call() {
        assert!(stop_after_first_for_request(true, Some(false)));
    }

    #[test]
    fn omitted_and_true_preserve_parallel_auto_calls() {
        assert!(!stop_after_first_for_request(true, None));
        assert!(!stop_after_first_for_request(true, Some(true)));
    }

    #[test]
    fn required_and_specific_choices_keep_the_existing_single_call_policy() {
        for parallel in [None, Some(false), Some(true)] {
            assert!(stop_after_first_for_request(false, parallel));
        }
    }

    #[test]
    fn auto_tool_calls_stay_unbiased_until_repeat_loop_threshold() {
        assert_eq!(super::tool_call_repeat_logit_bias(0), 0.0);
        assert_eq!(super::tool_call_repeat_logit_bias(1), 0.0);
        assert_eq!(super::tool_call_repeat_logit_bias(2), 0.0);
        assert_eq!(super::tool_call_repeat_logit_bias(3), -5.0);
        assert_eq!(super::tool_call_repeat_logit_bias(4), -10.0);
    }
}

#[cfg(test)]
mod request_deadline_tests {
    use super::checked_request_deadline;
    use super::contract_decode_deadline;

    #[test]
    fn decode_deadline_is_derived_only_when_it_exceeds_the_base() {
        let now = Instant::now();
        let secs = |d: Option<Instant>| d.map(|d| d.duration_since(now).as_secs());
        // 16384/8 + 10 + 30 = 2088 > base 300
        assert_eq!(
            secs(contract_decode_deadline(300, false, None, 16384, 2000, now)),
            Some(2088)
        );
        // small request: base already covers it, keep the pre-first-token deadline
        assert_eq!(
            contract_decode_deadline(300, false, None, 512, 100, now),
            None
        );
        // base 0, explicit request timeout and deepseek_v41: unchanged
        assert_eq!(
            contract_decode_deadline(0, false, None, 16384, 0, now),
            None
        );
        assert_eq!(
            contract_decode_deadline(300, false, Some(5.0), 16384, 0, now),
            None
        );
        assert_eq!(
            contract_decode_deadline(300, true, None, 16384, 0, now),
            None
        );
    }
    use std::time::{Duration, Instant};

    #[test]
    fn zero_still_disables_the_deadline() {
        let now = Instant::now();
        assert_eq!(checked_request_deadline(0.0, now), Ok(None));
        assert_eq!(checked_request_deadline(-0.0, now), Ok(None));
    }

    #[test]
    fn positive_integral_and_fractional_timeouts_keep_their_deadlines() {
        let now = Instant::now();
        for seconds in [0.125, 1.0, 300.0, 14340.0] {
            assert_eq!(
                checked_request_deadline(seconds, now),
                Ok(now.checked_add(Duration::from_secs_f32(seconds))),
            );
        }
    }

    #[test]
    fn negative_and_nonfinite_timeouts_are_errors_not_disabled_deadlines() {
        let now = Instant::now();
        for seconds in [-1.0, f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            assert!(checked_request_deadline(seconds, now).is_err());
        }
    }

    #[test]
    fn huge_finite_timeout_returns_an_error_without_panicking() {
        assert!(checked_request_deadline(f32::MAX, Instant::now()).is_err());
    }

    #[test]
    fn representable_duration_respects_the_platform_clock_range() {
        let seconds = 1.0e19_f32;
        let duration = Duration::try_from_secs_f32(seconds).unwrap();
        let now = Instant::now();
        match now.checked_add(duration) {
            None => assert!(checked_request_deadline(seconds, now).is_err()),
            Some(deadline) => {
                assert_eq!(checked_request_deadline(seconds, now), Ok(Some(deadline)));
            }
        }
    }
}

#[cfg(test)]
mod history_penalty_tests {
    use super::{HistoryPenalties, resolve_history_penalties};

    /// qwen3.6-27b's presets: presence plus frequency, DRY 0.8 and LZ 0.2.
    fn preset() -> atlas_kernels::SamplingCategory {
        atlas_kernels::SamplingCategory {
            presence_penalty: 1.0,
            frequency_penalty: 0.3,
            dry_multiplier: 0.8,
            lz_penalty: 0.2,
            ..atlas_kernels::SamplingPresets::default().non_thinking
        }
    }

    const NEUTRAL: HistoryPenalties = HistoryPenalties {
        presence: 0.0,
        frequency: 0.0,
        dry_multiplier: 0.0,
        lz: 0.0,
    };

    #[test]
    fn pinned_t0_neutralises_every_preset_history_penalty() {
        assert_eq!(
            resolve_history_penalties(Some(0.0), None, None, &preset()),
            NEUTRAL
        );
    }

    #[test]
    fn sampled_and_unpinned_requests_keep_the_preset() {
        let kept = HistoryPenalties {
            presence: 1.0,
            frequency: 0.3,
            dry_multiplier: 0.8,
            lz: 0.2,
        };
        // Unpinned covers ATLAS_IGNORE_REQUEST_SAMPLING and ATLAS_TEMP_OVERRIDE,
        // which serve the preset regime whatever the client sends.
        for temperature in [None, Some(0.6), Some(1.0)] {
            assert_eq!(
                resolve_history_penalties(temperature, None, None, &preset()),
                kept
            );
        }
    }

    #[test]
    fn client_penalties_still_apply_at_t0() {
        assert_eq!(
            resolve_history_penalties(Some(0.0), Some(0.5), Some(-0.25), &preset()),
            HistoryPenalties {
                presence: 0.5,
                frequency: -0.25,
                ..NEUTRAL
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_optional_filter;

    #[test]
    fn request_filter_outranks_model_and_server() {
        assert_eq!(resolve_optional_filter(Some(0.31), Some(0.17), 0.08), 0.31);
    }

    #[test]
    fn model_filter_outranks_server_and_preserves_explicit_zero() {
        assert_eq!(resolve_optional_filter(None, Some(0.0), 0.08), 0.0);
    }

    #[test]
    fn absent_request_and_model_filter_use_server_fallback() {
        assert_eq!(resolve_optional_filter(None, None, 0.08), 0.08);
    }
}

#[cfg(test)]
mod preset_selection_tests {
    use super::select_preset;

    fn presets() -> atlas_kernels::SamplingPresets {
        let mut presets = atlas_kernels::SamplingPresets::default();
        presets.thinking_text.temperature = 1.0;
        presets.thinking_coding.temperature = 0.9;
        presets.non_thinking.temperature = 0.0;
        presets.tools.temperature = 0.25;
        presets
    }

    #[test]
    fn default_policy_keeps_the_tools_preset_for_every_tool_turn() {
        let p = presets();
        for thinking in [false, true] {
            assert_eq!(select_preset(&p, true, thinking, false).temperature, 0.25);
        }
        assert_eq!(select_preset(&p, false, true, false).temperature, 1.0);
        assert_eq!(select_preset(&p, false, false, false).temperature, 0.0);
    }

    #[test]
    fn opted_in_models_sample_thinking_tool_turns_with_the_thinking_preset() {
        let p = presets();
        assert_eq!(select_preset(&p, true, true, true).temperature, 1.0);
        // Thinking-off tool turns and non-tool turns are unchanged.
        assert_eq!(select_preset(&p, true, false, true).temperature, 0.25);
        assert_eq!(select_preset(&p, false, true, true).temperature, 1.0);
        assert_eq!(select_preset(&p, false, false, true).temperature, 0.0);
    }
}
