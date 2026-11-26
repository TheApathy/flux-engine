// SPDX-License-Identifier: AGPL-3.0-only
//
// `MsgEntry` + the pre-loop builder that turns the inbound
// `ChatCompletionRequest.messages` into the local representation
// used by every downstream phase (json_messages, loop detector,
// task pin, observation mask, …).
//
// Lifted out of `chat::chat_completions_inner` (wave 4g) so the
// orchestrator stays under the 500-LoC cap.

use axum::http::StatusCode;
use axum::response::Response;
use std::sync::Arc;

use crate::AppState;
use crate::openai::ChatCompletionRequest;

use super::super::compact::openai_error_response;

/// Per-message data: role, content text, optional structured
/// `tool_calls`, and image-part count for the Jinja vision-marker
/// expansion. `pub(super)` so `chat::chat_completions_inner` and
/// the other `chat/*` sub-files can read every field.
pub(super) struct MsgEntry {
    pub(super) role: String,
    pub(super) content: String,
    /// Historical assistant reasoning trace forwarded to Jinja templates
    /// (enables `preserve_thinking: true` across multi-turn conversations).
    pub(super) reasoning_content: Option<String>,
    /// Structured tool_calls for the Jinja template (arguments
    /// pre-parsed to dicts).
    pub(super) tool_calls: Option<Vec<serde_json::Value>>,
    /// The caller's association with a historical tool invocation. Rendered
    /// into the Jinja message only for the native GLM contract (see
    /// `template::render_template`); other templates never see it.
    pub(super) tool_call_id: Option<String>,
    /// Number of image content parts on this message. When > 0
    /// the json_messages builder emits a structured content array
    /// so the Jinja template can render
    /// `<|vision_start|><|image_pad|><|vision_end|>` markers.
    pub(super) image_count: usize,
    /// Ordered UTF-8 byte positions in content, one per image marker.
    pub(super) image_text_offsets: Vec<usize>,
}

/// Outputs of [`build_msg_entries`]. Bundled as a struct because
/// the caller threads each field through five later phases.
pub(super) struct BuildOut {
    pub(super) messages: Vec<MsgEntry>,
    pub(super) cwd_hint: Option<String>,
    pub(super) image_pixels: Vec<(Vec<f32>, usize, usize)>,
    pub(super) image_pad_counts: Vec<usize>,
    pub(super) consecutive_tool_errors: u32,
    /// Number of historical tool-error bodies changed by recovery hints.
    pub(super) tool_error_hint_count: u32,
    /// Historical tool calls whose malformed JSON arguments were replaced.
    pub(super) invalid_tool_args_count: u32,
    /// Whether Atlas appended a normalized CWD block to the system prompt.
    pub(super) cwd_hint_injected: bool,
    /// Whether Atlas dropped a content-free leading system message.
    pub(super) vacuous_system_dropped: bool,
}

#[allow(clippy::result_large_err)]
pub(super) fn build_msg_entries(
    state: &Arc<AppState>,
    req: &ChatCompletionRequest,
    tools_active: bool,
) -> Result<BuildOut, Response> {
    let mut messages: Vec<MsgEntry> = Vec::with_capacity(req.messages.len());
    let mut all_images: Vec<String> = Vec::new();
    let mut image_pad_counts: Vec<usize> = Vec::new();
    let mut consecutive_tool_errors: u32 = 0;
    let mut tool_error_hint_count: u32 = 0;
    let mut invalid_tool_args_count: u32 = 0;

    let last_query_index = req
        .messages
        .iter()
        .rposition(|m| m.role == "user" && !m.content.text.starts_with("<tool_response>"))
        .unwrap_or(req.messages.len().saturating_sub(1));
    let uses_angle_think = state
        .think_start_token_id
        .and_then(|id| state.tokenizer.decode(&[id]).ok())
        .is_some_and(|tok| tok.trim() == "<think>");

    for (msg_idx, m) in req.messages.iter().enumerate() {
        let mut content = m.content.clone();
        content.validate_order().map_err(|e| {
            openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid message content: {e}"),
            )
        })?;
        // Opt-in legacy `<think>` history wrapper only when FLUX_AGENT_GUARDS=1
        // on `<think>`-based models when neither `reasoning_content` nor inline
        // `<think>` tags are already present. Modern Jinja templates own this
        // natively for Qwen3.5/3.8, GLM-5, Nemotron, Gemma 4, and Mistral.
        let thinking_suppressed = tools_active && !state.behavior.thinking_in_tools;
        if super::super::chat_phases::agent_guards_enabled()
            && m.role == "assistant"
            && state.tokenizer.supports_thinking()
            && uses_angle_think
            && msg_idx > last_query_index
            && !thinking_suppressed
            && m.reasoning_content.is_none()
            && !content.text.contains("<think>")
            && !content.text.contains("</think>")
        {
            content
                .prepend_text("<think>\n\n</think>\n\n")
                .map_err(|e| openai_error_response(StatusCode::BAD_REQUEST, e))?;
        }
        let mut text = content.text;

        // Preserve structured tool_calls for the Jinja template.
        // Always extract from assistant messages — past turns may
        // carry tool_calls that the template MUST render even when
        // the current request didn't pass `tools`.
        let tool_calls_json = if m.role == "assistant" {
            m.tool_calls.as_ref().and_then(|tcs| {
                if tcs.is_empty() {
                    return None;
                }
                let parsed: Vec<serde_json::Value> = tcs
                    .iter()
                    .map(|tc| {
                        let args: serde_json::Value =
                            match serde_json::from_str(&tc.function.arguments) {
                                Ok(args) => args,
                                Err(_) => {
                                    invalid_tool_args_count =
                                        invalid_tool_args_count.saturating_add(1);
                                    serde_json::Value::Object(Default::default())
                                }
                            };
                        serde_json::json!({
                            "id": tc.id.as_deref().unwrap_or(""),
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": args
                            }
                        })
                    })
                    .collect();
                Some(parsed)
            })
        } else {
            None
        };

        // Tool-response messages: pass raw content; Jinja template
        // handles `<tool_response>` wrapping and consecutive
        // grouping.
        if tools_active && m.role == "tool" {
            if crate::hint_injector::looks_like_error(&text) {
                consecutive_tool_errors += 1;
                if super::super::chat_phases::agent_guards_enabled() {
                    let original_len = text.len();
                    crate::hint_injector::inject_hints(&mut text, consecutive_tool_errors);
                    if text.len() != original_len {
                        tool_error_hint_count = tool_error_hint_count.saturating_add(1);
                    }
                }
            } else {
                consecutive_tool_errors = 0;
            }
        }

        let image_count = m.content.images.len();
        messages.push(MsgEntry {
            role: m.role.clone(),
            content: text,
            reasoning_content: m.reasoning_content.clone(),
            tool_calls: tool_calls_json,
            tool_call_id: m.tool_call_id.clone(),
            image_count,
            image_text_offsets: content.image_text_offsets,
        });
        if !m.content.images.is_empty() {
            for img_uri in &m.content.images {
                all_images.push(img_uri.clone());
                image_pad_counts.push(0);
            }
        }
    }

    // OpenAI 2025/2026 clients (SDK, Codex, Agents SDK) send `role: "developer"`
    // in place of or alongside `role: "system"`. Most chat templates (including
    // Qwen3.8) raise `Unexpected message role.` on `"developer"` and
    // `System message must be at the beginning.` on multiple leading system
    // messages. Normalize leading `developer` -> `system`, coalesce consecutive
    // leading text-only system blocks, and map mid-conversation `developer`
    // updates to `user`.
    normalize_developer_and_leading_system_roles(&mut messages);

    // Extract working directory from the system message if present.
    let cwd_hint: Option<String> = messages.iter().find(|m| m.role == "system").and_then(|m| {
        for line in m.content.lines() {
            let lower = line.to_lowercase();
            if (lower.contains("working directory")
                || lower.contains("working_directory")
                || lower.contains("cwd:"))
                && let Some(pos) = line.find(':')
            {
                let path = line[pos + 1..]
                    .trim()
                    .trim_matches(|c| c == '`' || c == '"' || c == '\'');
                if !path.is_empty() {
                    return Some(path.to_string());
                }
            }
        }
        None
    });

    // Inject CWD hint into the system message (NOT tool definitions —
    // those go to the Jinja template).
    let mut cwd_hint_injected = false;
    if tools_active
        && super::super::chat_phases::agent_guards_enabled()
        && let Some(ref cwd) = cwd_hint
    {
        let hints = format!("\n<environment>\nworking_directory: {cwd}\n</environment>");
        if let Some(first) = messages.first_mut()
            && first.role == "system"
        {
            first.content.push_str(&hints);
            cwd_hint_injected = true;
        }
    }

    // Neutralize a content-free leading system message. Clients (notably
    // Open WebUI's empty RAG/context template) inject a system message
    // carrying NO instruction — e.g. `"User Context:\n\n"` (trims to the
    // bare label `User Context:`). Models react to a content-free system
    // directive by producing terse / prematurely-terminated output
    // (isolated 2026-05-17: removing it 3x'd generation length on the
    // 3D-chess prompt). We can't fix the client, so Atlas adapts: treat
    // such a message as absent so a degenerate client prompt can't poison
    // generation. Conservative — only an empty body or a single short
    // bare `Label:` line qualifies; any substantive prompt is untouched.
    let vacuous_system_dropped = super::super::chat_phases::agent_guards_enabled()
        && messages.first().is_some_and(|m| {
            m.role == "system" && m.image_count == 0 && is_vacuous_system_content(&m.content)
        });
    if vacuous_system_dropped {
        let removed = messages.remove(0);
        tracing::info!(
            dropped = %removed.content.trim(),
            "Dropped content-free client system message (would bias the model toward terse output)"
        );
    }

    // Preprocess images if a vision config is available.
    let mut image_pixels: Vec<(Vec<f32>, usize, usize)> = Vec::new();
    if !all_images.is_empty() {
        let Some(vcfg) = &state.vision_config else {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                "Image input requires a vision-capable model".into(),
            ));
        };
        for (idx, uri) in all_images.iter().enumerate() {
            match spark_model::vision_preprocess::preprocess_image(uri, vcfg) {
                Ok((pixels, grid_h, grid_w)) => {
                    image_pad_counts[idx] = spark_model::vision_preprocess::image_pad_count(
                        grid_h,
                        grid_w,
                        vcfg.spatial_merge_size,
                    );
                    image_pixels.push((pixels, grid_h, grid_w));
                }
                Err(e) => {
                    return Err(openai_error_response(
                        StatusCode::BAD_REQUEST,
                        format!("Image decode error: {e}"),
                    ));
                }
            }
        }
    }
    Ok(BuildOut {
        messages,
        cwd_hint,
        image_pixels,
        image_pad_counts,
        consecutive_tool_errors,
        tool_error_hint_count,
        invalid_tool_args_count,
        cwd_hint_injected,
        vacuous_system_dropped,
    })
}

/// Normalize OpenAI `role: "developer"` messages so model chat templates that
/// only understand `system` / `user` / `assistant` / `tool` (e.g. Qwen3.8)
/// render them without raising `Unexpected message role.` or
/// `System message must be at the beginning.`.
///
/// - Leading `developer` messages become `system`.
/// - Consecutive leading text-only `system` / `developer` messages (e.g.
///   `[system, developer, user]`) coalesce into a single leading `system`
///   message joined with `"\n\n"`.
/// - Mid-conversation `developer` messages (after the first non-system/developer
///   turn) map to `user` so templates that forbid non-initial `system` blocks
///   keep functioning. Explicit mid-conversation `system` messages are left
///   untouched.
fn normalize_developer_and_leading_system_roles(messages: &mut Vec<MsgEntry>) {
    let leading_end = messages
        .iter()
        .take_while(|m| m.role == "system" || m.role == "developer")
        .count();

    let mut mid_start = leading_end;
    if leading_end > 0 {
        for m in &mut messages[..leading_end] {
            if m.role == "developer" {
                m.role = "system".to_string();
            }
        }
        // Coalesce consecutive leading text-only system messages into index 0.
        if leading_end > 1 && messages[..leading_end].iter().all(|m| m.image_count == 0) {
            let mut combined = String::new();
            for m in &messages[..leading_end] {
                if !m.content.is_empty() {
                    if !combined.is_empty() {
                        combined.push_str("\n\n");
                    }
                    combined.push_str(&m.content);
                }
            }
            messages[0].content = combined;
            messages.drain(1..leading_end);
            mid_start = 1;
        }
    }

    for m in &mut messages[mid_start..] {
        if m.role == "developer" {
            m.role = "user".to_string();
        }
    }
}

/// True when a system message carries no actual instruction and should
/// be treated as absent. Conservative by design — a substantive prompt
/// must never be stripped:
///   * empty / whitespace-only body, OR
///   * a single short bare label line ending in ':' with nothing after
///     it (e.g. `User Context:`, `Context:`, `System:`) — the residue
///     of an empty client template (Open WebUI's RAG/context block).
/// Anything multi-line, or with any text past the colon, is a real
/// prompt and returns false.
fn is_vacuous_system_content(content: &str) -> bool {
    let t = content.trim();
    if t.is_empty() {
        return true;
    }
    if !t.contains('\n') && t.len() <= 32 && t.ends_with(':') {
        let label = &t[..t.len() - 1];
        return !label.is_empty()
            && label
                .chars()
                .all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '_' || c == '-');
    }
    false
}

#[cfg(test)]
mod vacuous_system_tests {
    use super::is_vacuous_system_content;

    #[test]
    fn empty_or_whitespace_is_vacuous() {
        assert!(is_vacuous_system_content(""));
        assert!(is_vacuous_system_content("   \n\t  "));
    }

    #[test]
    fn open_webui_empty_context_residue_is_vacuous() {
        // The exact 2026-05-17 field artifact.
        assert!(is_vacuous_system_content("User Context:\n\n"));
        assert!(is_vacuous_system_content("Context:"));
        assert!(is_vacuous_system_content("  System:  "));
    }

    #[test]
    fn substantive_prompt_is_not_vacuous() {
        assert!(!is_vacuous_system_content(
            "User Context:\nThe user is a senior Rust engineer."
        ));
        assert!(!is_vacuous_system_content("You are a helpful assistant."));
        assert!(!is_vacuous_system_content("Always answer in French."));
        // Label-like but with a real payload after the colon.
        assert!(!is_vacuous_system_content("Role: expert chess coach"));
        // Long single line ending in ':' is unusual prose, not a bare
        // header — keep it (avoid false-strip).
        assert!(!is_vacuous_system_content(
            "Summarize the following transcript and then ask the user this:"
        ));
    }

    fn text_entry(role: &str, content: &str) -> super::MsgEntry {
        super::MsgEntry {
            role: role.to_string(),
            content: content.to_string(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            image_count: 0,
            image_text_offsets: Vec::new(),
        }
    }

    #[test]
    fn developer_role_normalizes_and_coalesces_leading_system_blocks() {
        let mut msgs = vec![
            text_entry("system", "Base policy."),
            text_entry("developer", "Output JSON only."),
            text_entry("user", "Hello"),
            text_entry("developer", "Updated env context."),
        ];
        super::normalize_developer_and_leading_system_roles(&mut msgs);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "Base policy.\n\nOutput JSON only.");
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].content, "Hello");
        assert_eq!(msgs[2].role, "user");
        assert_eq!(msgs[2].content, "Updated env context.");
    }

    #[test]
    fn single_leading_developer_role_maps_to_system() {
        let mut msgs = vec![
            text_entry("developer", "Be concise."),
            text_entry("user", "Ping"),
        ];
        super::normalize_developer_and_leading_system_roles(&mut msgs);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "Be concise.");
        assert_eq!(msgs[1].role, "user");
    }
}
