// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

//! `/v1/chat/completions` orchestrator.
//!
//! Coordinates the request lifecycle across focused submodules:
//!
//! - `msg_entry`      — `MsgEntry` + `build_msg_entries` (req →
//!                      tokenisable shape, image preprocessing,
//!                      cwd extraction)
//! - `loop_detect`    — generic loop / spinning detection +
//!                      task-pin re-anchor
//! - `thinking`       — `(enable_thinking, thinking_budget)`
//!                      resolution
//! - `template`       — JSON-message build, auto-compact,
//!                      Jinja apply, image-pad expand,
//!                      template-forced-thinking detection
//! - `sampling_setup` — preset / penalty / stop-token / grammar /
//!                      timeout / logprobs resolution

mod image_admission;
#[cfg(test)]
mod image_admission_tests;
mod loop_detect;
mod msg_entry;
pub(super) mod repair_json;
mod sampling_setup;
mod template;
mod thinking;

pub(crate) use thinking::{effective_spontaneous_think_budget, set_server_thinking_controls};

use crate::main_modules::model_host::CurrentModel;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;

use crate::AppState;
use crate::openai::ChatCompletionRequest;

use super::compact::openai_error_response;

pub async fn chat_completions(
    CurrentModel(state): CurrentModel,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    body: axum::body::Bytes,
) -> Response {
    // Parse the body ourselves (instead of using axum's `Json`
    // extractor) so the same bytes can feed both the deserialized
    // handler path and the `--dump` raw-capture path without
    // cloning the struct or cascading `Serialize` through every
    // request type.
    if let Err(err) = super::schema_check::precheck_raw_chat_body(&body) {
        return super::compact::openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            err.to_string(),
            Some("response_format.json_schema.schema"),
            Some("invalid_json_schema"),
        );
    }

    let mut req: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };

    // Admission: the one settings snapshot this request resolves from.
    let defaults = crate::runtime_settings::pin(&mut req, &state.runtime);
    req.apply_default_max_tokens(
        crate::openai::body_sets_max_tokens(&body),
        defaults.default_max_tokens,
    );

    // --dump: record the incoming request body verbatim.
    let dump_seq = state.dump_writer.as_ref().and_then(|d| {
        match serde_json::from_slice::<serde_json::Value>(&body) {
            Ok(v) => {
                let seq = d.next_seq();
                d.dump_request("/v1/chat/completions", seq, &v);
                Some(seq)
            }
            Err(_) => None,
        }
    });

    if state.dsv41 {
        // deepseek_v41 renders from the wire body (api/dsv41.rs); the body
        // already parsed as a ChatCompletionRequest, so it is valid JSON.
        req.raw_body = serde_json::from_slice(&body).ok().map(Arc::new);
    }

    chat_completions_inner(state, req_ctx, req, dump_seq).await
}

/// The prompt-affecting phases' result: what the sampling and dispatch
/// phases need, whichever renderer produced it.
pub(crate) struct PreparedChat {
    pub(crate) tools_active: bool,
    pub(crate) cwd_hint: Option<String>,
    pub(crate) image_pixels: Vec<(Vec<f32>, usize, usize)>,
    pub(crate) prompt_tokens: Vec<u32>,
    pub(crate) enable_thinking: bool,
    pub(crate) thinking_budget: Option<u32>,
}

/// Parser prompts are a fallback for templates that do not receive tool
/// definitions.  Qwen's native XML Jinja already renders the schemas and wire
/// instructions from `tools`; prepending `Qwen3CoderParser::system_prompt()` a
/// second time gives the model two competing copies of the protocol.  Ordinary
/// Hermes-model templates retain their established behavior; a Hermes override
/// on a native-Qwen template is rejected separately because historical calls
/// would still render as Qwen XML. TSCG still needs the full parser prompt
/// because it deliberately withholds `tools` from Jinja.
fn should_inject_parser_system_prompt(
    parser_name: &str,
    native_qwen_xml_jinja_receives_tools: bool,
) -> bool {
    !(parser_name == "qwen3_coder" && native_qwen_xml_jinja_receives_tools)
}

fn incompatible_native_qwen_parser(
    native_qwen_xml_template: bool,
    parser_name: &str,
) -> Option<String> {
    (native_qwen_xml_template && parser_name != "qwen3_coder").then(|| {
        format!(
            "tool-call parser '{parser_name}' is incompatible with this checkpoint's native Qwen XML chat template; use qwen3_coder"
        )
    })
}

fn tool_system_prompt(
    parser: &dyn crate::tool_parser::ToolCallParser,
    tools: &[crate::tool_parser::ToolDefinition],
    tool_choice: &crate::tool_parser::ToolChoice,
    native_qwen_xml_jinja_receives_tools: bool,
) -> String {
    if should_inject_parser_system_prompt(parser.name(), native_qwen_xml_jinja_receives_tools) {
        parser.system_prompt(tools, tool_choice)
    } else {
        // Qwen4's native Jinja already owns the one schema/protocol copy. Keep
        // only Atlas's behavioral delta here (immediate use, no narration or
        // fake results, and required/specific choice steering).
        crate::tool_parser::qwen3_coder_native_safety_prompt(tool_choice)
    }
}

/// Internal entry for the parsed-request path. Called by
/// [`chat_completions`] after body capture, and by the Responses
/// API adapter (which builds a `ChatCompletionRequest` in-memory
/// and skips HTTP body bytes). `dump_seq` is `Some` only on the
/// public handler path.
pub(crate) async fn chat_completions_inner(
    state: Arc<AppState>,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    mut req: ChatCompletionRequest,
    dump_seq: Option<u64>,
) -> Response {
    crate::metrics::REQUESTS_TOTAL.inc();
    // Guarded, not a bare `inc()`: the ten early returns below all end the
    // request, and every one of them used to leak a count.
    let active = crate::metrics::ActiveRequestGuard::new();
    // The entry point pinned it already; internal callers pin here.
    let defaults = crate::runtime_settings::pin(&mut req, &state.runtime);
    if let Some(mct) = req.max_completion_tokens {
        req.max_tokens = mct;
    }

    // ── Input validation + cross-turn F-feature guards ──
    if let Err(resp) = super::chat_phases::validate_input(&req) {
        return resp;
    }
    if let Err(message) = req.validate_rendered_prompt_evidence_nonce() {
        return openai_error_response(StatusCode::BAD_REQUEST, message.to_string());
    }
    // Native GLM tool contract admission (no-op for every other parser).
    if let Err(message) = crate::tool_parser::request_admission::capability(
        req.tools.as_deref().unwrap_or(&[]),
        req.tool_choice.as_ref(),
        state.tool_call_parser.as_deref(),
        state.behavior.disable_tool_grammar,
    ) {
        return openai_error_response(StatusCode::BAD_REQUEST, message);
    }
    for message in &req.messages {
        if let Err(error) = message.content.validate_order() {
            return openai_error_response(StatusCode::BAD_REQUEST, error);
        }
    }
    let image_count = req.messages.iter().fold(0usize, |count, m| {
        count.saturating_add(m.content.images.len())
    });
    if let Err(error) = image_admission::validate(
        image_count,
        state.vision_config.is_some() || state.dsv41_vision.is_some(),
        state.max_batch_size,
        state.yarn_context,
    ) {
        return openai_error_response(StatusCode::BAD_REQUEST, error.into());
    }
    if state.dsv41 {
        if req.rendered_prompt_evidence_nonce.is_some() {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                "rendered prompt evidence is not supported on the DeepSeek wire renderer"
                    .to_string(),
            );
        }
        // deepseek_v41: the production Python server's prompt (api/dsv41.rs).
        // None of the generic request rewriting below (failure guards, tool
        // system prompt, observation masking, loop detection) exists there.
        let prepared = match super::dsv41::prepare(&state, &req) {
            Ok((p, max_tokens_cap)) => {
                // app.py clamps max_tokens to the context left over.
                req.max_tokens = req.max_tokens.min(max_tokens_cap);
                p
            }
            Err(resp) => return resp,
        };
        return sample_and_dispatch(state, req_ctx, req, dump_seq, active, prepared, false, 0)
            .await;
    }
    let f23_metrics = if super::chat_phases::agent_guards_enabled() {
        super::chat_phases::apply_failure_guards(&mut req)
    } else {
        super::failures::F23ProgressMetrics {
            score: 0,
            attempts: 0,
        }
    };
    let _ = f23_metrics; // kept available for downstream consumers

    // Tool-active gating.
    let tools_active = state.tool_call_parser.is_some()
        && req.tools.as_ref().is_some_and(|t| !t.is_empty())
        && !req.tool_choice.as_ref().is_some_and(|tc| tc.is_none());

    // A native Qwen template renders not only the current tool definitions but
    // also prior assistant tool_calls/tool results in Qwen XML. Merely hiding
    // the current definitions cannot make a Hermes (or other) override safe in
    // a multi-turn request: its prompt/grammar would conflict with the history
    // bytes. Reject the incompatible combination before any prompt is rendered.
    if tools_active
        && let Some(ref parser) = state.tool_call_parser
        && let Some(message) =
            incompatible_native_qwen_parser(state.tokenizer.native_qwen_xml_tools(), parser.name())
    {
        return openai_error_response(StatusCode::BAD_REQUEST, message);
    }

    // Inject the parser prompt for non-native templates, or opt-in Atlas
    // safety delta / explicit tool_choice steering for native Qwen XML.
    // On native Qwen XML with `tool_choice="auto"` and default public parity
    // (`FLUX_AGENT_GUARDS` off), leave the system prompt untouched.
    if tools_active && let Some(ref parser) = state.tool_call_parser {
        let default_choice = crate::tool_parser::ToolChoice::Mode("auto".to_string());
        let tool_choice = req.tool_choice.as_ref().unwrap_or(&default_choice);
        let native_qwen_xml_jinja_receives_tools = template::should_pass_tools_to_jinja(
            tools_active,
            crate::tscg::tscg_enabled(),
            state.tokenizer.native_qwen_xml_tools(),
            Some(parser.name()),
        ) && state.tokenizer.native_qwen_xml_tools();
        let skip_native_safety_auto = native_qwen_xml_jinja_receives_tools
            && !super::chat_phases::agent_guards_enabled()
            && matches!(tool_choice, crate::tool_parser::ToolChoice::Mode(m) if m == "auto");
        if !skip_native_safety_auto {
            let tool_prompt = tool_system_prompt(
                parser.as_ref(),
                req.tools.as_deref().unwrap_or(&[]),
                tool_choice,
                native_qwen_xml_jinja_receives_tools,
            );
            if let Some(first) = req.messages.first_mut().filter(|m| m.role == "system") {
                if let Err(e) = first.content.prepend_text(&format!("{tool_prompt}\n\n")) {
                    return super::compact::openai_error_response(StatusCode::BAD_REQUEST, e);
                }
            } else {
                req.messages.insert(
                    0,
                    crate::openai::IncomingMessage::synthetic_system(tool_prompt),
                );
            }
        }
    }

    tracing::info!(
        "Request: model={}, messages={}, tools={}, tools_active={}, tool_choice={:?}, stream={}, temp={:?}, max_tokens={}, freq_pen={:?}, rep_pen={:?}",
        req.model,
        req.messages.len(),
        req.tools.as_ref().map_or(0, |t| t.len()),
        tools_active,
        req.tool_choice,
        req.stream,
        req.temperature,
        req.max_tokens,
        req.frequency_penalty,
        req.repetition_penalty,
    );

    // ── Phase 1: build MsgEntry vec + image preprocess + cwd ────
    let msg_entry::BuildOut {
        mut messages,
        cwd_hint,
        image_pixels,
        image_pad_counts,
        consecutive_tool_errors,
        tool_error_hint_count,
        invalid_tool_args_count,
        cwd_hint_injected,
        vacuous_system_dropped,
    } = match msg_entry::build_msg_entries(&state, &req, tools_active) {
        Ok(o) => o,
        Err(resp) => return resp,
    };
    if req.report_interventions {
        for _ in 0..tool_error_hint_count {
            req.record_intervention("tool_error_hint", "append_recovery_hint");
        }
        for _ in 0..invalid_tool_args_count {
            req.record_intervention("tool_args", "replace_invalid_json_with_empty_object");
        }
    }
    if cwd_hint_injected {
        req.record_intervention("cwd_context", "append_environment_hint");
    }
    if vacuous_system_dropped {
        req.record_intervention("vacuous_system", "drop_system_message");
    }

    // ── Phase 1.5: merge server-level chat_template_kwargs default ─
    // Merge only unset fields from `--default-chat-template-kwargs` so a
    // client's `preserve_thinking` (or other explicit kwargs) is never
    // overwritten by the server default.
    if let Some(ref default_kw) = state.default_chat_template_kwargs {
        let thinking_explicit = req.thinking_explicitly_requested();
        let mut kw =
            req.chat_template_kwargs
                .clone()
                .unwrap_or(crate::openai::ChatTemplateKwargs {
                    enable_thinking: None,
                    thinking_budget: None,
                    preserve_thinking: None,
                });
        let mut changed = req.chat_template_kwargs.is_some();
        if !thinking_explicit {
            if !defaults.thinking_default_runtime
                && kw.enable_thinking.is_none()
                && default_kw.enable_thinking.is_some()
            {
                kw.enable_thinking = default_kw.enable_thinking;
                changed = true;
            }
            if kw.thinking_budget.is_none() && default_kw.thinking_budget.is_some() {
                kw.thinking_budget = default_kw.thinking_budget;
                changed = true;
            }
        }
        if kw.preserve_thinking.is_none() && default_kw.preserve_thinking.is_some() {
            kw.preserve_thinking = default_kw.preserve_thinking;
            changed = true;
        }
        if changed {
            req.chat_template_kwargs = Some(kw);
        }
    }

    // ── Phase 2: thinking resolution (pre-template) ─────────────
    let (enable_thinking, thinking_budget) =
        thinking::resolve_thinking(&state, &defaults, &mut req, tools_active);

    // ── Phase 3: stale-failure observation masking (opt-in agent guards) ─
    if super::chat_phases::agent_guards_enabled() {
        let bodies: Vec<(&str, &str)> = messages
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str()))
            .collect();
        let mask = crate::observation_mask::compute_masking(&bodies, 2);
        let mut masked_count = 0usize;
        for (i, replacement) in mask.into_iter().enumerate() {
            if let Some(new_body) = replacement
                && messages[i].image_count == 0
            {
                messages[i].content = new_body;
                masked_count += 1;
            }
        }
        if masked_count > 0 {
            tracing::info!(
                masked_count,
                "observation_mask: elided {masked_count} stale tool-failure bodies"
            );
            crate::metrics::OBSERVATION_MASK_ELIDED_BODIES.inc_by(masked_count as u64);
            if req.report_interventions {
                for _ in 0..masked_count {
                    req.record_intervention("observation_mask", "elide_stale_tool_failure");
                }
            }
        }
    }

    // ── Phase 4: generic loop / spinning detection + task pin ───
    let loop_detect::LoopDetectOut {
        suppress_tool_call,
        tool_call_repeat_count,
        loop_hint_injected,
        task_pin_anchor_count,
    } = loop_detect::check_loops(&req, &mut messages, consecutive_tool_errors, tools_active);
    if suppress_tool_call {
        req.record_intervention("loop_detector", "suppress_tool_call");
    }
    if loop_hint_injected {
        req.record_intervention("loop_detector", "append_progress_hint");
    }
    if req.report_interventions {
        for _ in 0..task_pin_anchor_count {
            req.record_intervention("task_pin", "append_goal_reminder");
        }
    }

    // ── Phase 5: render Jinja template + image-pad expansion ────
    let template::TemplateOut {
        prompt_tokens,
        enable_thinking,
        thinking_budget,
        xml_leak_stripped_count,
        auto_compacted,
        overflow_truncated_count,
        template_forced_thinking,
    } = match template::render_template(
        &state,
        &req,
        &messages,
        &image_pad_counts,
        enable_thinking,
        thinking_budget,
        tools_active,
    ) {
        Ok(o) => o,
        Err(resp) => return resp,
    };
    if req.report_interventions {
        for _ in 0..xml_leak_stripped_count {
            req.record_intervention("xml_leak_guard", "strip_historical_tool_markup");
        }
    }
    if auto_compacted {
        req.record_intervention("auto_compact", "rewrite_history");
    }
    if req.report_interventions {
        for _ in 0..overflow_truncated_count {
            req.record_intervention("context_overflow", "drop_history_message");
        }
    }
    if template_forced_thinking {
        req.record_intervention("template", "force_thinking");
    }

    let prepared = PreparedChat {
        tools_active,
        cwd_hint,
        image_pixels,
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    };
    sample_and_dispatch(
        state,
        req_ctx,
        req,
        dump_seq,
        active,
        prepared,
        suppress_tool_call,
        tool_call_repeat_count,
    )
    .await
}

#[cfg(test)]
mod native_tool_prompt_tests {
    use super::template::tool_definitions_for_jinja;
    use super::{
        incompatible_native_qwen_parser, should_inject_parser_system_prompt, tool_system_prompt,
    };
    use crate::tool_parser::{
        FunctionDefinition, Qwen3CoderParser, ToolCallParser, ToolChoice, ToolChoiceFunction,
        ToolDefinition,
    };

    fn probe_tool() -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "Bash".into(),
                description: Some("Run a command".into()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                })),
            },
        }
    }

    fn render_native_prompt(tool_choice: &ToolChoice) -> String {
        let parser = Qwen3CoderParser;
        let tool = probe_tool();
        let safety = tool_system_prompt(&parser, std::slice::from_ref(&tool), tool_choice, true);
        let messages = vec![
            serde_json::json!({"role": "system", "content": safety}),
            serde_json::json!({"role": "user", "content": "use the command tool"}),
        ];
        let tools: Vec<serde_json::Value> =
            tool_definitions_for_jinja(&[tool], true, Some("qwen3_coder"))
                .iter()
                .map(|tool| serde_json::to_value(tool).unwrap())
                .collect();

        // The production override path (`load_override_template` + `build_jinja_env`),
        // not a bare Environment: the template's `.startswith` / `.endswith` only
        // render after that rewrite.
        let env = crate::tokenizer::override_template_env_for_tests(include_str!(
            "../../../../../jinja-templates/qwen3_5_moe.jinja"
        ))
        .unwrap();
        env.get_template("chat")
            .unwrap()
            .render(minijinja::context! {
                messages => minijinja::Value::from_serialize(&messages),
                tools => minijinja::Value::from_serialize(&tools),
                add_generation_prompt => true,
                enable_thinking => false,
                preserve_thinking => false,
                disable_tool_steering => false,
                add_vision_id => false,
            })
            .unwrap()
    }

    #[test]
    fn qwen_native_jinja_renders_one_tool_schema_not_two() {
        let rendered = render_native_prompt(&ToolChoice::Mode("auto".into()));

        assert_eq!(rendered.matches("\"name\"").count(), 1, "{rendered}");
        assert_eq!(rendered.matches("# Tools").count(), 1, "{rendered}");
        assert_eq!(rendered.matches("[atlas-f33]").count(), 1, "{rendered}");
        assert_eq!(
            rendered.matches("<ATLAS_TOOL_SAFETY>").count(),
            1,
            "{rendered}"
        );
        assert!(
            rendered.contains("emit the function call directly"),
            "{rendered}"
        );
        assert!(
            rendered.contains("do not pre-render them as prose"),
            "{rendered}"
        );
    }

    #[test]
    fn native_qwen_safety_delta_preserves_required_and_specific_choice() {
        let required = render_native_prompt(&ToolChoice::Mode("required".into()));
        assert!(required.contains("must call at least one offered function"));
        assert_eq!(required.matches("# Tools").count(), 1);

        let specific = render_native_prompt(&ToolChoice::Specific {
            function: ToolChoiceFunction {
                name: "Bash".into(),
            },
        });
        assert!(specific.contains("must call the 'Bash' function"));
        assert_eq!(specific.matches("# Tools").count(), 1);
    }

    #[test]
    fn native_qwen_rejects_non_native_parser_before_multiturn_history_render() {
        let prior_native_history = serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "Bash", "arguments": {"command": "pwd"}}
            }]
        });
        assert!(prior_native_history["tool_calls"].is_array());
        assert!(
            incompatible_native_qwen_parser(true, "hermes")
                .unwrap()
                .contains("use qwen3_coder")
        );
        assert!(incompatible_native_qwen_parser(true, "qwen3_coder").is_none());
        assert!(incompatible_native_qwen_parser(false, "hermes").is_none());
    }

    #[test]
    fn older_qwen_tscg_and_regular_hermes_models_keep_parser_prompt_fallback() {
        assert!(should_inject_parser_system_prompt("qwen3_coder", false));
        assert!(should_inject_parser_system_prompt("hermes", true));
        assert!(should_inject_parser_system_prompt("hermes", false));
    }
}

/// Phases 6-7, shared by every renderer: context check, sampling preset /
/// stop / grammar / timeout, then the streaming or blocking dispatch.
#[allow(clippy::too_many_arguments)]
async fn sample_and_dispatch(
    state: Arc<AppState>,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    req: ChatCompletionRequest,
    dump_seq: Option<u64>,
    active: crate::metrics::ActiveRequestGuard,
    prepared: PreparedChat,
    suppress_tool_call: bool,
    tool_call_repeat_count: usize,
) -> Response {
    let mut req = req;
    let defaults = crate::runtime_settings::pin(&mut req, &state.runtime);
    let PreparedChat {
        tools_active,
        cwd_hint,
        image_pixels,
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    } = prepared;
    let intervention_ledger = req
        .report_interventions
        .then(|| std::sync::Arc::new(std::sync::Mutex::new(req.intervention_ledger.clone())));
    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let tools_count = req.tools.as_ref().map_or(0, |t| t.len());
    tracing::info!(
        "Session {session_hash:#x}: {prompt_tokens} prompt tokens, tools={tools_active} ({tools_count} defined)",
        prompt_tokens = prompt_tokens.len()
    );
    let prompt_len = prompt_tokens.len();
    if prompt_len >= state.max_seq_len {
        // Overflow truncation (see `template.rs`) defaults to off; reaching here
        // means the rendered prompt exceeds `max_seq_len` (or, when opt-in
        // overflow truncation is enabled, even the minimal tail cannot fit —
        // e.g. a single oversized system prompt or user turn). Emit the
        // OpenAI-standard `context_length_exceeded` code so coding-agent clients
        // (OpenCode/Cline) compact and retry instead of blindly re-sending the
        // same over-long prompt.
        return super::compact::openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            format!(
                "Prompt too long: {prompt_len} tokens exceeds max_seq_len {} (leave room for output tokens)",
                state.max_seq_len
            ),
            Some("messages"),
            Some("context_length_exceeded"),
        );
    }
    // The digest is computed after final prompt construction from the exact
    // token vector that will be dispatched. Raw decoded prompt bytes remain
    // request-local and are never returned or logged.
    let rendered_prompt_evidence_source =
        if let Some(nonce) = req.rendered_prompt_evidence_nonce.clone() {
            let prompt = match state.tokenizer.decode_with_special(&prompt_tokens) {
                Ok(prompt) => prompt,
                Err(error) => {
                    tracing::error!("failed to decode prompt for evidence digest: {error:#}");
                    return openai_error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Failed to compute rendered prompt evidence".to_string(),
                    );
                }
            };
            let digest = crate::openai::RenderedPromptEvidence::digest(&prompt);
            drop(prompt);
            Some((nonce, digest))
        } else {
            None
        };

    // ── Phase 6: sampling preset / stop / grammar / timeout ─────
    let sampling_setup::SamplingSetup {
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
    } = match sampling_setup::build_sampling(
        &state,
        &defaults,
        &req,
        enable_thinking,
        tools_active,
        suppress_tool_call,
        tool_call_repeat_count,
    ) {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    // A server-defaulted allowance shrinks to the context left after the
    // prompt rather than failing admission; the thinking budget, sized
    // against the default, is re-capped to the smaller allowance.
    let (max_tokens, thinking_budget) = if req.max_tokens_defaulted {
        let allowance = super::context_budget::defaulted_output_allowance(
            max_tokens,
            prompt_len,
            state.max_seq_len,
        );
        let thinking_budget = if allowance < max_tokens {
            thinking_budget.map(|budget| {
                thinking::recap_thinking_budget(
                    budget,
                    allowance,
                    tools_active,
                    state.behavior.thinking_in_tools,
                    defaults.thinking_answer_reserve,
                )
            })
        } else {
            thinking_budget
        };
        (allowance, thinking_budget)
    } else {
        (max_tokens, thinking_budget)
    };
    // One line per request with what was actually applied after request,
    // MODEL.toml preset and policy resolution, so a tester can see the
    // effective thinking budget and sampling instead of only the client's fields.
    tracing::info!(
        "Effective request: thinking={enable_thinking}, thinking_budget={thinking_budget:?}, \
         reasoning_effort={:?}, temperature={temperature}, top_p={top_p}, top_k={top_k}, \
         min_p={min_p}, presence_penalty={presence_penalty}, \
         repetition_penalty={repetition_penalty}, max_tokens={max_tokens}{}",
        req.requested_reasoning_effort(),
        defaults.log_suffix(),
    );

    // ── Request-lifecycle contract: admission sized to the context left ──
    let (max_tokens, thinking_budget, contract_warning) =
        match admit_request_output(&state, prompt_len, max_tokens) {
            Ok(None) => (max_tokens, thinking_budget, None),
            Ok(Some((clamped, warning))) => {
                let thinking_budget = thinking_budget.map(|budget| {
                    thinking::recap_thinking_budget(
                        budget,
                        clamped,
                        tools_active,
                        state.behavior.thinking_in_tools,
                        defaults.thinking_answer_reserve,
                    )
                });
                (clamped, thinking_budget, Some(warning))
            }
            Err(resp) => return resp,
        };
    let (max_tokens, thinking_budget) = thinking::apply_thinking_reserve(
        enable_thinking,
        req.thinking_reserve,
        max_tokens,
        thinking_budget,
        prompt_len,
        state.max_seq_len,
    );
    // `timeout_at` (the server base or the explicit request timeout) bounds
    // the request until its first token, prefill included; the derived
    // decode deadline replaces it from the first token on.
    let decode_timeout_at = sampling_setup::contract_decode_deadline(
        state.request_timeout,
        state.dsv41,
        req.timeout,
        max_tokens,
        prompt_len,
        std::time::Instant::now(),
    );

    // ── Phase 7: dispatch streaming or blocking ─────────────────
    if req.stream {
        // The streaming path owns the decrement from here.
        active.release();
        let response = super::chat_stream_dispatch::dispatch_streaming(
            state,
            &req,
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
            logit_bias.clone(),
            enable_thinking,
            thinking_budget,
            tools_active,
            tool_choice_required,
            suppress_tool_call,
            cwd_hint.clone(),
            stop_tokens,
            grammar_spec.clone(),
            top_logprobs,
            timeout_at,
            decode_timeout_at,
            contract_warning.clone().into_iter().collect(),
            intervention_ledger,
            rendered_prompt_evidence_source,
        )
        .await;
        return with_contract_warning(response, contract_warning.as_deref());
    }

    // As does the blocking path.
    active.release();
    let warning_header = contract_warning.clone();
    let response =
        super::chat_blocking::run_blocking_path(super::chat_blocking::BlockingPathArgs {
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
            warnings: contract_warning.into_iter().collect(),
        })
        .await;
    with_contract_warning(response, warning_header.as_deref())
}

/// Admission for `prompt + max_tokens` against `max_seq_len`: `Ok(None)`
/// when it fits, `Ok(Some((clamped, warning)))` when `max_tokens` was
/// lowered to the room left, `Err` with the OpenAI error envelope otherwise.
#[allow(clippy::result_large_err)]
fn admit_request_output(
    state: &AppState,
    prompt_len: usize,
    max_tokens: usize,
) -> Result<Option<(usize, String)>, Response> {
    use super::request_contract::{
        AdmissionConfig, OutputAdmission, admit_output, clamp_warning, too_little_room_message,
    };
    let cfg = AdmissionConfig::current();
    let max_seq_len = state.max_seq_len;
    match admit_output(&cfg, prompt_len, max_tokens, max_seq_len) {
        OutputAdmission::Fits => Ok(None),
        OutputAdmission::Clamped { from, to } => {
            tracing::warn!(
                "MAX_TOKENS_CLAMPED from={from} to={to} prompt={prompt_len} max_seq_len={max_seq_len}"
            );
            Ok(Some((to, clamp_warning(from, to, prompt_len, max_seq_len))))
        }
        OutputAdmission::TooLittleRoom { room } => {
            Err(super::compact::openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                too_little_room_message(
                    prompt_len,
                    max_tokens,
                    max_seq_len,
                    room,
                    cfg.min_answer_tokens,
                ),
                Some("messages"),
                Some("context_length_exceeded"),
            ))
        }
        OutputAdmission::Exceeds => Err(super::compact::openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            format!(
                "Context too long: {prompt_len} prompt + {max_tokens} requested output tokens exceeds max_seq_len {max_seq_len}"
            ),
            Some("messages"),
            Some("context_length_exceeded"),
        )),
    }
}

/// Header carrying a request-lifecycle warning, on streaming and blocking
/// responses alike (the blocking body also lists it under `warnings`).
pub(crate) const CONTRACT_WARNING_HEADER: &str = "x-flux-warning";

fn with_contract_warning(mut response: Response, warning: Option<&str>) -> Response {
    if let Some(value) = warning.and_then(|w| axum::http::HeaderValue::from_str(w).ok()) {
        response
            .headers_mut()
            .insert(CONTRACT_WARNING_HEADER, value);
    }
    response
}

#[cfg(test)]
mod contract_warning_tests {
    use super::*;

    #[test]
    fn a_clamp_warning_is_attached_as_a_header_and_absent_otherwise() {
        let plain = with_contract_warning(StatusCode::OK.into_response(), None);
        assert!(plain.headers().get(CONTRACT_WARNING_HEADER).is_none());
        let warned = with_contract_warning(
            StatusCode::OK.into_response(),
            Some("max_tokens_clamped: x"),
        );
        assert_eq!(
            warned
                .headers()
                .get(CONTRACT_WARNING_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("max_tokens_clamped: x")
        );
    }
}
