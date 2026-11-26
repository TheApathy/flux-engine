// SPDX-License-Identifier: AGPL-3.0-only
//
// Chat-template rendering: build the JSON-message array, run the
// optional auto-compact, apply the Jinja template, expand image
// pad-tokens, and detect template-forced thinking.
//
// Lifted out of `chat::chat_completions_inner` (wave 4g).

use axum::http::StatusCode;
use axum::response::Response;
use std::sync::Arc;

use crate::AppState;
use crate::openai::{ChatCompletionRequest, ParsedContent};
use crate::tool_parser;

use super::super::compact::{compact_messages, openai_error_response, truncate_to_fit};
use super::super::failures::strip_xml_leaks_from_assistant_content;
use super::msg_entry::MsgEntry;

/// Whether the hard context-overflow truncation safety net is active.
/// Default OFF so client history is never silently dropped (standard
/// OpenAI/vLLM 400-on-overflow contract); opt in with
/// `FLUX_CTX_OVERFLOW_TRUNCATE=1` (or legacy `ATLAS_CTX_OVERFLOW_TRUNCATE=1`).
/// Cached so the hot path is a relaxed load.
fn ctx_overflow_truncate_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let val = std::env::var("FLUX_CTX_OVERFLOW_TRUNCATE")
            .ok()
            .or_else(|| std::env::var("ATLAS_CTX_OVERFLOW_TRUNCATE").ok());
        matches!(val.as_deref(), Some("1" | "true" | "on"))
    })
}

pub(super) fn should_pass_tools_to_jinja(
    tools_active: bool,
    tscg_enabled: bool,
    native_qwen_xml_template: bool,
    parser_name: Option<&str>,
) -> bool {
    should_pass_tools_to_jinja_with_suppression(
        tools_active,
        tscg_enabled,
        native_qwen_xml_template,
        parser_name,
        parser_name == Some("bare_json"),
    )
}

fn should_pass_tools_to_jinja_with_suppression(
    tools_active: bool,
    tscg_enabled: bool,
    native_qwen_xml_template: bool,
    parser_name: Option<&str>,
    parser_suppresses_jinja_tools: bool,
) -> bool {
    if !tools_active || tscg_enabled || parser_suppresses_jinja_tools {
        return false;
    }
    // The request path rejects this incompatible native-Qwen override
    // before rendering. Keep this fail-closed check too: feeding tools to
    // Qwen XML Jinja while another parser owns the prompt/grammar would
    // create contradictory protocols.
    !native_qwen_xml_template || parser_name == Some("qwen3_coder")
}

pub(super) fn tool_definitions_for_jinja(
    tools: &[tool_parser::ToolDefinition],
    native_qwen_xml_template: bool,
    parser_name: Option<&str>,
) -> Vec<tool_parser::ToolDefinition> {
    if native_qwen_xml_template && parser_name == Some("qwen3_coder") {
        // The native template owns schema/protocol rendering. Preserve the
        // parser's F33 Bash retry guard by mutating the one schema copy that
        // Jinja receives instead of prepending a second parser-owned schema.
        tool_parser::qwen3_coder_tools_with_atlas_safety(tools)
    } else {
        tools.to_vec()
    }
}

/// Outputs of [`render_template`]. Threaded into the streaming /
/// blocking dispatch.
pub(super) struct TemplateOut {
    pub(super) prompt_tokens: Vec<u32>,
    /// Possibly overridden by template-forced-thinking detection.
    pub(super) enable_thinking: bool,
    pub(super) thinking_budget: Option<u32>,
    /// Historical assistant messages whose leaked tool XML was stripped.
    pub(super) xml_leak_stripped_count: u32,
    /// Whether the opt-in progressive compactor changed prompt history.
    pub(super) auto_compacted: bool,
    /// Number of historical messages dropped by overflow recovery.
    pub(super) overflow_truncated_count: u32,
    /// Whether the rendered template overrode a disabled thinking request.
    pub(super) template_forced_thinking: bool,
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)]
pub(super) fn render_template(
    state: &Arc<AppState>,
    req: &ChatCompletionRequest,
    messages: &[MsgEntry],
    image_pad_counts: &[usize],
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    tools_active: bool,
) -> Result<TemplateOut, Response> {
    // The thinking ceiling of this request's admission snapshot
    // (`/v1/flux/settings`); the startup value when nothing pinned one.
    let max_thinking_budget = req
        .runtime_defaults
        .as_ref()
        .map_or(state.behavior.max_thinking_budget, |d| {
            d.max_thinking_budget
        });
    // Use closed thinking when client doesn't explicitly enable it.
    let template_thinking = enable_thinking;
    let image_count: usize = messages.iter().map(|m| m.image_count).sum();
    let has_images = image_count != 0;
    if image_count != image_pad_counts.len() || image_pad_counts.contains(&0) {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            "Image markers and preprocessed images must have matching nonzero counts".into(),
        ));
    }

    // Build JSON messages with structured tool_calls for Jinja.
    let stripper_tools: &[tool_parser::ToolDefinition] = req.tools.as_deref().unwrap_or(&[]);
    // GLM's template orders parallel tool results by `tool_call_id`. Only the
    // native GLM contract forwards it, so no other template's input changes.
    let forward_tool_call_id = state
        .tool_call_parser
        .as_ref()
        .is_some_and(|parser| parser.name() == "glm_xml");
    let mut xml_leak_stripped_count = 0u32;
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            let effective_content = if m.role == "assistant"
                && m.tool_calls.as_ref().is_some_and(|tcs| !tcs.is_empty())
                && !stripper_tools.is_empty()
                && m.image_count == 0
            {
                let stripped = strip_xml_leaks_from_assistant_content(&m.content, stripper_tools);
                if stripped != m.content {
                    xml_leak_stripped_count = xml_leak_stripped_count.saturating_add(1);
                }
                stripped
            } else {
                m.content.clone()
            };
            let content_val = ParsedContent::marker_json(
                &effective_content,
                m.image_count,
                &m.image_text_offsets,
            )?;
            let mut msg = serde_json::json!({"role": m.role, "content": content_val});
            if let Some(ref rc) = m.reasoning_content {
                msg["reasoning_content"] = serde_json::Value::String(rc.clone());
            }
            if let Some(ref tcs) = m.tool_calls {
                msg["tool_calls"] = serde_json::Value::Array(tcs.clone());
            }
            if forward_tool_call_id && let Some(ref id) = m.tool_call_id {
                msg["tool_call_id"] = serde_json::Value::String(id.clone());
            }
            Ok::<_, String>(msg)
        })
        .collect::<Result<_, _>>()
        .map_err(|e| openai_error_response(StatusCode::BAD_REQUEST, e))?;
    // When TSCG is enabled or the parser suppresses Jinja tools (e.g.
    // BareJsonParser), the parser's `system_prompt()` has already placed the
    // tool schema into messages[0]; passing `tools` to Jinja as well would
    // re-render the template's native tool block and defeat compaction or
    // inject contradictory tool instructions. Pass `None` so the template's
    // `{% if tools %}` branch falls through.
    let parser_suppresses = state
        .tool_call_parser
        .as_ref()
        .is_some_and(|parser| parser.suppresses_jinja_tools());
    let jinja_tools: Option<Vec<serde_json::Value>> = if should_pass_tools_to_jinja_with_suppression(
        tools_active,
        crate::tscg::tscg_enabled(),
        state.tokenizer.native_qwen_xml_tools(),
        state.tool_call_parser.as_ref().map(|parser| parser.name()),
        parser_suppresses,
    ) {
        req.tools.as_ref().map(|ts| {
            let defs = if super::super::chat_phases::agent_guards_enabled() {
                tool_definitions_for_jinja(
                    ts,
                    state.tokenizer.native_qwen_xml_tools(),
                    state.tool_call_parser.as_ref().map(|parser| parser.name()),
                )
            } else {
                ts.to_vec()
            };
            defs.iter()
                .map(|t| serde_json::to_value(t).unwrap_or_default())
                .collect()
        })
    } else {
        None
    };

    // Progressive auto-compact (disabled by default; opt in via `--auto-compact`).
    let auto_compact_threshold = state.auto_compact_threshold.filter(|&t| t > 0.0);
    let auto_compact_active = auto_compact_threshold.is_some();
    // Pixel ownership is request-wide: do not drop/rewrite image-bearing history.
    let mut auto_compacted = false;
    let json_messages = if !has_images && auto_compact_active && json_messages.len() > 4 {
        let threshold = auto_compact_threshold.expect("auto_compact_active implies Some");
        let trial_tokens = state
            .tokenizer
            .apply_chat_template_openai_with_options(
                &json_messages,
                jinja_tools.as_deref(),
                template_thinking,
                state.behavior.disable_tool_steering,
                req.requested_reasoning_effort(),
                req.chat_template_kwargs
                    .as_ref()
                    .and_then(|kw| kw.preserve_thinking)
                    .or(state.behavior.preserve_thinking),
            )
            .map(|t| t.len())
            .unwrap_or(0);
        let trigger_tokens = (state.max_seq_len as f32 * threshold.clamp(0.01, 1.0)) as usize;
        if trial_tokens > trigger_tokens {
            let compacted = compact_messages(&json_messages, trial_tokens, state.max_seq_len);
            auto_compacted = compacted != json_messages;
            compacted
        } else {
            json_messages
        }
    } else {
        json_messages
    };

    let tokenize = |jm: &[serde_json::Value]| {
        state.tokenizer.apply_chat_template_openai_with_options(
            jm,
            jinja_tools.as_deref(),
            template_thinking,
            state.behavior.disable_tool_steering,
            req.requested_reasoning_effort(),
            req.chat_template_kwargs
                .as_ref()
                .and_then(|kw| kw.preserve_thinking)
                .or(state.behavior.preserve_thinking),
        )
    };
    let mut prompt_tokens = match tokenize(&json_messages) {
        Ok(t) => t,
        Err(e) => {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Tokenization error: {e}"),
            ));
        }
    };

    // ── Optional context-overflow truncation safety net ──
    // Disabled by default (strict 400-on-overflow). Opt in via
    // `ATLAS_CTX_OVERFLOW_TRUNCATE=1` to reserve output headroom and drop the
    // oldest turns when a multi-turn conversation would overflow `max_seq_len`.
    let mut overflow_truncated_count = 0u32;
    if !has_images && ctx_overflow_truncate_enabled() {
        let output_reserve = overflow_output_reserve(
            max_thinking_budget,
            super::thinking::generation_max_tokens(
                req.max_tokens,
                tools_active,
                state.tool_max_tokens,
            ),
        );
        let fit_budget = state.max_seq_len.saturating_sub(output_reserve).max(1);
        if prompt_tokens.len() >= fit_budget {
            let mut jm = json_messages.clone();
            // keep_tail shrinks each pass (6 → 4 → 2) so we always make progress.
            for keep_tail in [6usize, 4, 2] {
                match truncate_to_fit(&jm, keep_tail) {
                    Some(truncated) => {
                        let dropped = jm.len().saturating_sub(truncated.len());
                        jm = truncated;
                        match tokenize(&jm) {
                            Ok(t) => {
                                overflow_truncated_count = overflow_truncated_count
                                    .saturating_add(u32::try_from(dropped).unwrap_or(u32::MAX));
                                let fits = t.len() < fit_budget;
                                tracing::warn!(
                                    "ctx-overflow truncate (keep_tail={keep_tail}): dropped {dropped} \
                                     msg(s), prompt {} → {} tokens (fit_budget={fit_budget}, fits={fits})",
                                    prompt_tokens.len(),
                                    t.len(),
                                );
                                prompt_tokens = t;
                                if fits {
                                    break;
                                }
                            }
                            Err(e) => {
                                tracing::error!("re-tokenization after truncate failed: {e}");
                                break;
                            }
                        }
                    }
                    None => break, // nothing left to drop → caller fast-fails with 400
                }
            }
        }
    }

    // The image pad the model's vision path consumes: the checkpoint's
    // `image_pad_token_id` when its vision config declares one (GLM-5.3 uses
    // <|image|>, not Qwen's <|image_pad|>), else the tokenizer's Qwen marker.
    let image_pad_id = state
        .vision_config
        .as_ref()
        .map(|vision| vision.image_pad_token_id)
        .filter(|&token| token != 0)
        .or_else(|| state.tokenizer.image_pad_token_id());
    // Refuse templates that omit/duplicate images, before expansion can hide the mismatch.
    if has_images {
        let pad = image_pad_id.ok_or_else(|| {
            openai_error_response(
                StatusCode::BAD_REQUEST,
                "Tokenizer lacks an image marker token".into(),
            )
        })?;
        // GLM-5.3's published template is text-only: for each image it renders
        // a fixed "unable to process this image" reminder. The checkpoint is
        // multimodal, so that exact tokenized span is replaced by the native
        // <|begin_of_image|><|image|><|end_of_image|> triplet. A template that
        // already emits pads is left alone.
        if prompt_tokens.iter().all(|&id| id != pad) {
            prompt_tokens = materialize_glm_image_placeholders(
                &state.tokenizer,
                prompt_tokens,
                image_count,
                pad,
            )
            .map_err(|e| {
                openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Image template error: {e:#}"),
                )
            })?;
        }
        if prompt_tokens.iter().filter(|&&id| id == pad).count() != image_count {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                "Chat template did not preserve exactly one marker per image".into(),
            ));
        }
    }
    // Expand image pads when needed.
    let prompt_tokens = match image_pad_id {
        Some(pad) if image_pad_counts.iter().any(|&c| c > 1) => {
            expand_image_pads_with_id(prompt_tokens, pad, image_pad_counts)
        }
        _ => prompt_tokens,
    };

    // Template-forced thinking detection.
    let mut template_forced_thinking = false;
    let (enable_thinking, thinking_budget) = if let Some(think_start) = state.think_start_token_id {
        let tail = &prompt_tokens[prompt_tokens.len().saturating_sub(8)..];
        let last_start = tail.iter().rposition(|t| *t == think_start);
        let has_unclosed_think = match (last_start, state.think_end_token_id) {
            (Some(si), Some(end_tok)) => !tail[si + 1..].contains(&end_tok),
            (Some(_), None) => true,
            (None, _) => false,
        };
        if has_unclosed_think && !enable_thinking {
            template_forced_thinking = true;
            let mt = u32::try_from(super::thinking::generation_max_tokens(
                req.max_tokens,
                tools_active,
                state.tool_max_tokens,
            ))
            .unwrap_or(u32::MAX);
            let answer_reserve = req
                .runtime_defaults
                .as_ref()
                .map_or(state.behavior.thinking_answer_reserve, |d| {
                    d.thinking_answer_reserve
                });
            let off_ceiling = match super::thinking::effective_thinking_off_budget() {
                Some(off) if max_thinking_budget > 0 => off.min(max_thinking_budget),
                Some(off) => off,
                None => max_thinking_budget,
            };
            let capped_budget = super::thinking::cap_thinking_budget(
                None,
                off_ceiling,
                mt,
                tools_active,
                state.behavior.thinking_in_tools,
                answer_reserve,
            );
            tracing::info!(
                "Template-forced thinking detected (unclosed \\<think\\> in prompt tail) — \
                 overriding enable_thinking=true with budget={}",
                capped_budget,
            );
            (true, Some(capped_budget))
        } else {
            (enable_thinking, thinking_budget)
        }
    } else {
        (enable_thinking, thinking_budget)
    };

    Ok(TemplateOut {
        prompt_tokens,
        enable_thinking,
        thinking_budget,
        xml_leak_stripped_count,
        auto_compacted,
        overflow_truncated_count,
        template_forced_thinking,
    })
}

/// GLM-5.3 image placeholders (ported from perf/glm-5.3-flash-prefill-20260920).
fn materialize_glm_image_placeholders(
    tokenizer: &crate::tokenizer::ChatTokenizer,
    tokens: Vec<u32>,
    image_count: usize,
    pad_id: u32,
) -> anyhow::Result<Vec<u32>> {
    let existing = tokens.iter().filter(|&&token| token == pad_id).count();
    if existing == image_count {
        return Ok(tokens);
    }
    anyhow::ensure!(
        existing == 0,
        "rendered prompt has {existing} image pads for {image_count} images"
    );
    let reminder = tokenizer.encode(
        "<reminder>You are unable to process this image because you don't have multi-modal input ability. Try different methods.</reminder>",
    )?;
    anyhow::ensure!(!reminder.is_empty(), "GLM image reminder encoded empty");
    let singleton = |text: &str| -> anyhow::Result<u32> {
        let encoded = tokenizer.encode(text)?;
        anyhow::ensure!(
            encoded.len() == 1,
            "GLM media marker {text} must encode to one token"
        );
        Ok(encoded[0])
    };
    let replacement = [
        singleton("<|begin_of_image|>")?,
        pad_id,
        singleton("<|end_of_image|>")?,
    ];
    replace_exact_spans(tokens, &reminder, &replacement, image_count)
}

fn expand_image_pads_with_id(tokens: Vec<u32>, pad_id: u32, counts: &[usize]) -> Vec<u32> {
    let extra: usize = counts.iter().map(|count| count.saturating_sub(1)).sum();
    let mut output = Vec::with_capacity(tokens.len() + extra);
    let mut image = 0usize;
    for token in tokens {
        if token == pad_id {
            output.extend(std::iter::repeat_n(
                pad_id,
                counts.get(image).copied().unwrap_or(1).max(1),
            ));
            image += 1;
        } else {
            output.push(token);
        }
    }
    output
}

fn replace_exact_spans(
    tokens: Vec<u32>,
    needle: &[u32],
    replacement: &[u32],
    expected: usize,
) -> anyhow::Result<Vec<u32>> {
    anyhow::ensure!(
        !needle.is_empty(),
        "image placeholder needle must not be empty"
    );
    let mut output = Vec::with_capacity(tokens.len());
    let mut cursor = 0usize;
    let mut replaced = 0usize;
    while cursor < tokens.len() {
        if tokens[cursor..].starts_with(needle) {
            output.extend_from_slice(replacement);
            cursor += needle.len();
            replaced += 1;
        } else {
            output.push(tokens[cursor]);
            cursor += 1;
        }
    }
    anyhow::ensure!(
        replaced == expected,
        "rendered prompt contained {replaced} GLM image reminders for {expected} images"
    );
    Ok(output)
}

/// Output room the overflow safety net keeps free: the thinking ceiling, but
/// never more than this request can emit, plus 256 tokens of answer slack.
/// Reserving a large model ceiling that the request cannot reach would drop
/// conversation history that still fits.
fn overflow_output_reserve(max_thinking_budget: u32, generation_max: usize) -> usize {
    (max_thinking_budget as usize)
        .min(generation_max)
        .saturating_add(256)
}

#[cfg(test)]
mod overflow_output_reserve_tests {
    use super::overflow_output_reserve;

    #[test]
    fn reserve_is_the_thinking_ceiling_when_the_request_can_reach_it() {
        assert_eq!(overflow_output_reserve(2048, 4096), 2304);
        assert_eq!(overflow_output_reserve(2048, 2048), 2304);
    }

    #[test]
    fn reserve_never_exceeds_what_the_request_can_emit() {
        assert_eq!(overflow_output_reserve(32768, 4096), 4352);
        assert_eq!(overflow_output_reserve(32768, 0), 256);
    }
}

#[cfg(test)]
mod native_qwen_tool_render_tests {
    use super::{should_pass_tools_to_jinja, tool_definitions_for_jinja};
    use crate::tool_parser::{FunctionDefinition, ToolDefinition};

    fn bash_tool() -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "Bash".into(),
                description: Some("Run a shell command".into()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                })),
            },
        }
    }

    #[test]
    fn native_qwen_and_parser_override_never_render_competing_protocols() {
        assert!(should_pass_tools_to_jinja(
            true,
            false,
            true,
            Some("qwen3_coder")
        ));
        assert!(!should_pass_tools_to_jinja(
            true,
            false,
            true,
            Some("hermes")
        ));
        assert!(should_pass_tools_to_jinja(
            true,
            false,
            false,
            Some("hermes")
        ));
        assert!(!should_pass_tools_to_jinja(
            true,
            true,
            true,
            Some("qwen3_coder")
        ));
    }

    #[test]
    fn parser_suppressing_jinja_tools_withholds_template_tools() {
        use crate::tool_parser::{BareJsonParser, HermesParser, ToolCallParser};

        assert!(BareJsonParser.suppresses_jinja_tools());
        assert!(!HermesParser.suppresses_jinja_tools());
        assert!(!super::should_pass_tools_to_jinja_with_suppression(
            true,
            false,
            false,
            Some(BareJsonParser.name()),
            BareJsonParser.suppresses_jinja_tools(),
        ));
        assert!(!should_pass_tools_to_jinja(
            true,
            false,
            false,
            Some(BareJsonParser.name()),
        ));
        assert!(super::should_pass_tools_to_jinja_with_suppression(
            true,
            false,
            false,
            Some(HermesParser.name()),
            HermesParser.suppresses_jinja_tools(),
        ));
    }

    #[test]
    fn native_qwen_jinja_schema_retains_f33_exactly_once() {
        let guarded = tool_definitions_for_jinja(&[bash_tool()], true, Some("qwen3_coder"));
        let rendered_schema = serde_json::to_string(&guarded).unwrap();

        assert_eq!(rendered_schema.matches("[atlas-f33]").count(), 1);
        assert!(rendered_schema.contains("command not found"));

        let unchanged = tool_definitions_for_jinja(&[bash_tool()], false, Some("qwen3_coder"));
        assert!(
            !serde_json::to_string(&unchanged)
                .unwrap()
                .contains("[atlas-f33]")
        );
    }
}
