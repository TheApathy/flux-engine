// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Provenance for output parsing that is not part of the configured parser's
/// canonical wire protocol.
///
/// The API layer deliberately decides how to publish this count (for example,
/// as intervention-ledger events). Keeping the counter here makes recovery
/// provenance travel with parsing without coupling the parser to HTTP types.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ToolParseAudit {
    pub recovery_count: usize,
}

impl ToolParseAudit {
    pub fn recovered(self) -> bool {
        self.recovery_count != 0
    }
}

/// Parse completed output and report whether the configured parser needed a
/// non-canonical recovery path.
///
/// Qwen's canonical publication grammar is an exact structural envelope:
/// `<tool_call><function=NAME>` followed by fully closed
/// `<parameter=KEY>...</parameter>` elements and `</function>`. The terminal
/// outer `</tool_call>` alone may be absent because a serving stop boundary can
/// consume it. Alternate spellings, foreign formats, malformed inner closures,
/// and parser-side argument deletion are recoveries.
pub fn parse_tool_calls_with_audit(
    text: &str,
    expected_parser: &str,
) -> (Option<String>, Vec<ToolCall>, ToolParseAudit) {
    let (content, calls) = parse_tool_calls_impl(text);
    let recovery_count =
        if expected_parser == "qwen3_coder" && !canonical_qwen_envelopes_match(text, calls.len()) {
            1
        } else {
            0
        };
    (content, calls, ToolParseAudit { recovery_count })
}

/// Return true only for the canonical Qwen inner wire grammar. Whitespace
/// between structural elements is protocol formatting; tag spellings,
/// function/parameter names, and all inner closing tags are exact.
pub(super) fn is_canonical_qwen_body(text: &str) -> bool {
    let Some(mut rest) = text.trim_start().strip_prefix("<function=") else {
        return false;
    };
    let Some(name_end) = rest.find('>') else {
        return false;
    };
    let name = &rest[..name_end];
    if name != name.trim() || !is_tool_name_component(name) {
        return false;
    }
    rest = &rest[name_end + 1..];

    let mut parameter_names = std::collections::HashSet::new();
    loop {
        rest = rest.trim_start();
        if let Some(after_close) = rest.strip_prefix("</function>") {
            return after_close.trim().is_empty();
        }

        let Some(after_open) = rest.strip_prefix("<parameter=") else {
            return false;
        };
        let Some(key_end) = after_open.find('>') else {
            return false;
        };
        let key = &after_open[..key_end];
        if key != key.trim()
            || !is_tool_name_component(key)
            || key == "_think"
            || !parameter_names.insert(key)
        {
            return false;
        }

        let value_and_tail = &after_open[key_end + 1..];
        let Some(value_end) = value_and_tail.find("</parameter>") else {
            return false;
        };
        // A new parameter or function close before the proper parameter close
        // is precisely the missing-close recovery accepted by the tolerant
        // parser, not canonical Qwen syntax.
        if value_and_tail[..value_end].contains("<parameter=")
            || value_and_tail[..value_end].contains("</function>")
        {
            return false;
        }
        rest = &value_and_tail[value_end + "</parameter>".len()..];
    }
}

fn canonical_qwen_envelopes_match(text: &str, parsed_calls: usize) -> bool {
    let text = glm_xml_scan::after_thinking(text);
    let mut rest = text;
    let mut canonical_calls = 0usize;
    let mut outside_envelopes = String::new();

    while let Some(start) = rest.find("<tool_call>") {
        outside_envelopes.push_str(&rest[..start]);
        outside_envelopes.push('\n');
        let body = &rest[start + "<tool_call>".len()..];
        if let Some(end) = glm_xml_scan::legacy_close(body) {
            if !is_canonical_qwen_body(&body[..end]) {
                return false;
            }
            canonical_calls = canonical_calls.saturating_add(1);
            rest = &body[end + glm_xml_scan::CLOSE.len()..];
        } else {
            // A serving stop can remove only the terminal outer close. The
            // complete inner function/parameter grammar must still be present.
            if !is_canonical_qwen_body(body) {
                return false;
            }
            canonical_calls = canonical_calls.saturating_add(1);
            rest = "";
        }
    }
    outside_envelopes.push_str(rest);

    canonical_calls == parsed_calls && !qwen_residual_text_requires_recovery(&outside_envelopes)
}

/// Report foreign tool-call syntax that appears outside canonical Qwen
/// envelopes without constructing a [`ToolCall`] or consuming a call ID.
///
/// The streaming detector retains the exact wire transcript and calls this at
/// terminal flush. That closes the mixed-format gap where a canonical call was
/// emitted first and a later bare `name{json}` sibling had already flowed out
/// as ordinary content before the legacy fallbacks ran.
pub(super) fn qwen_residual_requires_recovery(text: &str) -> bool {
    let text = glm_xml_scan::after_thinking(text);
    let mut rest = text;
    let mut outside_envelopes = String::new();
    while let Some(start) = rest.find("<tool_call>") {
        outside_envelopes.push_str(&rest[..start]);
        outside_envelopes.push('\n');
        let body = &rest[start + "<tool_call>".len()..];
        if let Some(end) = glm_xml_scan::legacy_close(body) {
            rest = &body[end + glm_xml_scan::CLOSE.len()..];
        } else {
            // A missing terminal outer close makes the rest part of the final
            // envelope, not residual content.
            rest = "";
        }
    }
    outside_envelopes.push_str(rest);
    qwen_residual_text_requires_recovery(&outside_envelopes)
}

fn qwen_residual_text_requires_recovery(outside_envelopes: &str) -> bool {
    // The main parser intentionally stops its foreign-format fallbacks once a
    // canonical envelope produced a call. Audit the residual text separately
    // so a canonical call cannot hide a Mistral, bare-function, MiniMax,
    // Gemma, tools-wrapper, or JSON sibling that serving left in content. This
    // detector is lexical and side-effect-free: invoking a fallback parser
    // here would consume global tool-call IDs merely because audit is enabled.
    let foreign_marker = [
        MISTRAL_TOOL_CALLS_TAG,
        "<function=",
        "<function ",
        "<|function=",
        "<|function ",
        "<minimax:tool_call>",
        "<minimax:_call>",
        "<|tool_call>",
        "<tools>",
        "<invoke name=",
        "<parameter=",
        "\ncall:",
        "\n_call:",
    ]
    .iter()
    .any(|marker| outside_envelopes.contains(marker));
    let json_tool_shape =
        outside_envelopes.contains("\"name\"") && outside_envelopes.contains("\"arguments\"");
    foreign_marker || json_tool_shape || contains_bare_identifier_json_call(outside_envelopes)
}

/// Parse tool calls from completed model output.
///
/// Scans for `<tool_call></tool_call>` tags and auto-detects inner format
/// (JSON for hermes, XML for qwen3_coder).
///
/// Returns `(content, tool_calls)` where content is text outside tags.
pub fn parse_tool_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    parse_tool_calls_impl(text)
}

fn parse_tool_calls_impl(text: &str) -> (Option<String>, Vec<ToolCall>) {
    // Strip <think>...</think> before parsing tool calls (matches vLLM behavior).
    // Tool calls inside thinking blocks are model deliberation, not real invocations.
    let text = glm_xml_scan::after_thinking(text);
    // MiniMax uses `<minimax:tool_call>` as the outer wrapper (different
    // tag from Qwen's `<tool_call>`). Normalize both wrappers to the
    // same outer form so the scanning loop below doesn't need
    // per-parser branches. Allocation only when the namespaced tag
    // actually appears.
    let owned_normalized: String;
    let text: &str = if text.contains("<minimax:tool_call>")
        || text.contains("</minimax:tool_call>")
        // F71 (2026-04-29): also accept the live-observed broken
        // BPE-merge variants (`<minimax:_call>` / `</minimax:_call>`)
        // and a few other near-miss forms. xgrammar's TagDispatch is
        // non-anchored across BPE merge boundaries, so the model can
        // emit `<minimax:_call>` instead of `<minimax:tool_call>`
        // without the matcher catching it. If the inner
        // `<invoke name="...">…</invoke>` block is still intact, we
        // can salvage a valid tool_call by normalising the broken
        // outer envelope back to the canonical form.
        || text.contains("<minimax:_call>")
        || text.contains("</minimax:_call>")
    {
        owned_normalized = glm_xml_scan::normalize_minimax(text);
        owned_normalized.as_str()
    } else {
        text
    };
    let mut calls = Vec::new();
    let mut content_parts = Vec::new();
    let mut rest = text;
    let mut idx = 0u32;
    let mut native_seen = false;

    loop {
        match rest.find("<tool_call>") {
            Some(start) => {
                let before = rest[..start].trim();
                if !before.is_empty() {
                    content_parts.push(before.to_string());
                }
                rest = &rest[start + 11..];
                if glm_xml_scan::native_prefix(rest) {
                    native_seen = true;
                    if let Some(end) = glm_xml_scan::native_close(rest) {
                        if let Some(call) = glm_xml::parse_body(&rest[..end]) {
                            calls.push(call);
                            idx += 1;
                        } else {
                            content_parts.push(format!("<tool_call>{}</tool_call>", &rest[..end]));
                        }
                        rest = &rest[end + glm_xml_scan::CLOSE.len()..];
                        continue;
                    }
                    content_parts.push(format!("<tool_call>{rest}"));
                    break; // Never salvage an unclosed native envelope.
                }
                match glm_xml_scan::legacy_close(rest) {
                    Some(end) => {
                        let parsed = parse_envelope_calls(rest[..end].trim(), idx);
                        idx += parsed.len() as u32;
                        calls.extend(parsed);
                        rest = &rest[end + 12..];
                    }
                    None => {
                        // No closing </tool_call> — likely truncated by max_tokens.
                        // Try to parse the truncated content as tool call(s) anyway.
                        // The JSON may be incomplete, but parse_one_call handles this.
                        let parsed = parse_envelope_calls(rest.trim(), idx);
                        if !parsed.is_empty() {
                            calls.extend(parsed);
                        } else {
                            content_parts.push(format!("<tool_call>{rest}"));
                        }
                        break;
                    }
                }
            }
            None => {
                let after = rest.trim();
                if !after.is_empty() {
                    content_parts.push(after.to_string());
                }
                break;
            }
        }
    }
    if native_seen {
        // Native values and malformed native envelopes are not input to any
        // foreign-format salvage pass, even when zero valid calls were found.
        let content = (!content_parts.is_empty()).then(|| content_parts.join("\n"));
        return (content, calls);
    }
    // Fallback 00: Gemma-4 bare `fn_name{key:val,...}` at text start (no
    // wrapper). The gemma4.jinja tool-steering prefix injects
    // `<|tool_call>call:` into the prompt, so model output begins directly
    // with `fn_name{...}` — parse_bare_identifier_json_calls rejects because
    // keys are unquoted and strings use `<|"|>`. Apply gemma4_to_json
    // conversion first, then attempt JSON parse.
    //
    // Narrowly gated: text must START with an identifier directly followed
    // by `{`, so prose won't match. If successful, trailing content after
    // the balanced `}` is discarded (model often spews garbage tokens after
    // the call).
    if calls.is_empty() {
        let mut trimmed = text.trim_start();
        // Strip leading `namespace:` prefixes like `google:google_search{...}`
        // that Gemma-4-26B-A4B NVFP4A16 produces when its weakened
        // instruction-following lets pretraining priors ("google search"
        // canonical phrase) override the declared tool name. The trailing
        // identifier after the colon is the part that should match a tool
        // name (exactly or fuzzily via downstream logic).
        if let Some(colon) = trimmed.find(':') {
            let head = &trimmed[..colon];
            let is_ident = !head.is_empty()
                && head
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.');
            if is_ident && !head.starts_with("call") && !head.starts_with("_call") {
                // Only strip when there's another identifier followed by `{`
                // after the colon — keeps things like `call:fn{...}` for
                // the existing Gemma-4 native-format path untouched.
                let rest = &trimmed[colon + 1..];
                let rest_id_end = rest
                    .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '.')
                    .unwrap_or(rest.len());
                if rest_id_end >= 2 && rest.as_bytes().get(rest_id_end) == Some(&b'{') {
                    trimmed = rest;
                }
            }
        }
        let id_end = trimmed
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '.')
            .unwrap_or(trimmed.len());
        if id_end >= 2
            && trimmed.as_bytes().get(id_end) == Some(&b'{')
            && trimmed.as_bytes()[0].is_ascii_alphabetic()
        {
            let name = trimmed[..id_end].to_string();
            let args_part = &trimmed[id_end..];
            if let Some(end_rel) = find_balanced_json_end(args_part) {
                let json_slice = &args_part[..end_rel];
                let converted = gemma4_to_json(json_slice);
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&converted)
                    && v.is_object()
                {
                    let args = serde_json::to_string(&v).unwrap_or_else(|_| "{}".into());
                    calls.push(ToolCall {
                        id: next_tool_call_id(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name,
                            arguments: args,
                        },
                    });
                    return (None, calls);
                }
            }
        }
    }

    // Fallback 0a: Mistral native format `[TOOL_CALLS]name[ARGS]{json}`.
    // Must run before the Gemma-4 and JSON fallbacks so Mistral's bracketed
    // tokens don't get misparsed as bare JSON.
    if calls.is_empty() && text.contains(MISTRAL_TOOL_CALLS_TAG) {
        let (m_content, m_calls) = parse_mistral_native_calls(text);
        if !m_calls.is_empty() {
            return (m_content, m_calls);
        }
    }

    // Fallback 0: Gemma-4 native format `<|tool_call>call:fn{...}<tool_call|>`
    // Must run before `parse_bare_identifier_json_calls` so Gemma-4 tool calls
    // with JSON arguments aren't stolen with leaked `<|tool_call>call:` delimiters.
    if calls.is_empty() {
        let mut g4_rest = text;
        let mut g4_content = Vec::new();
        loop {
            match g4_rest.find("<|tool_call>") {
                Some(start) => {
                    let before = g4_rest[..start].trim();
                    if !before.is_empty() {
                        g4_content.push(before.to_string());
                    }
                    g4_rest = &g4_rest[start + 12..]; // len("<|tool_call>") = 12
                    let end = g4_rest.find("<tool_call|>").unwrap_or(g4_rest.len());
                    let inner = g4_rest[..end].trim();
                    if let Some(tc) = parse_gemma4_native_call(inner) {
                        calls.push(tc);
                    }
                    g4_rest = if end < g4_rest.len() {
                        &g4_rest[end + 12..]
                    } else {
                        ""
                    };
                }
                None => {
                    let after = g4_rest.trim();
                    if !after.is_empty() {
                        g4_content.push(after.to_string());
                    }
                    break;
                }
            }
        }
        if !calls.is_empty() {
            let content = if g4_content.is_empty() {
                None
            } else {
                Some(g4_content.join("\n"))
            };
            return (content, calls);
        }
    }

    // Fallback 0b: bare Mistral-style `fn_name{json}` without [TOOL_CALLS]/[ARGS].
    // Pass-8 regression: Mistral-Small-4 NVFP4 emits e.g.
    //   `I'll check the current weather in Paris for you.get_weather{"city": "Paris"}`
    // The model dropped the [TOOL_CALLS] prefix in its quantized form but still
    // produces identifier-{json}. Scan for identifier patterns followed by a
    // balanced JSON object and promote them to tool calls.
    if calls.is_empty() {
        let (bc_content, bc_calls) = parse_bare_identifier_json_calls(text);
        if !bc_calls.is_empty() {
            return (bc_content, bc_calls);
        }
    }

    // Fallback 1: look for <tools>JSON</tools> wrapper (some models use this
    // instead of <tool_call>).
    if calls.is_empty() {
        let (tools_content, tools_calls) = parse_tools_tag_calls(text);
        if !tools_calls.is_empty() {
            return (tools_content, tools_calls);
        }
    }

    // Fallback 2a: bare `call:fn{...}` without <|tool_call> wrapper (Gemma-4 NVFP4).
    if calls.is_empty() {
        let trimmed = text.trim();
        if trimmed.starts_with("call:")
            || trimmed.starts_with("_call:")
            || trimmed.contains("\ncall:")
            || trimmed.contains("\n_call:")
        {
            let mut bare_content = Vec::new();
            for line in trimmed.split('\n') {
                let line = line.trim();
                if line.starts_with("call:") || line.starts_with("_call:") {
                    if let Some(tc) = parse_gemma4_native_call(line) {
                        calls.push(tc);
                    }
                } else if !line.is_empty() {
                    bare_content.push(line.to_string());
                }
            }
            if !calls.is_empty() {
                let content = if bare_content.is_empty() {
                    None
                } else {
                    Some(bare_content.join("\n"))
                };
                return (content, calls);
            }
        }
    }

    // Fallback 2b: look for bare <function> or <function= tags without <tool_call> wrapper.
    // Models at lower quantization sometimes omit the wrapper tags.
    if calls.is_empty() {
        let (bare_content, bare_calls) = parse_bare_function_calls(text);
        if !bare_calls.is_empty() {
            return (bare_content, bare_calls);
        }
    }

    // Fallback 2c: BARE `<invoke name="X">…<parameter name="K">V</parameter>…</invoke>`
    // blocks without any envelope (no `<tool_call>`, no `<minimax:tool_call>`).
    // Triggered by Qwen3.6 cross-format contamination — observed
    // 2026-05-09 OpenClaw stress run where the model issued 5 well-formed
    // qwen3_coder envelopes then switched mid-response to MiniMax-style
    // bare `<invoke>` blocks. The non-streaming chat_blocking path has
    // no separate salvage hook (unlike chat_stream's tool_salvage),
    // so we recover here using the existing `parse_minimax_xml_calls_all`
    // — same function the streaming detector uses for the inner body
    // of a MiniMax envelope.
    if calls.is_empty() && text.contains("<invoke name=") {
        let bare_invoke_calls = super::parse_minimax_xml_calls_all(text);
        if !bare_invoke_calls.is_empty() {
            // Strip the recovered invoke blocks from content so the
            // client doesn't see both the structured tool_calls and
            // the literal XML in `content`.
            let mut clean = text.to_string();
            let mut search = 0usize;
            while let Some(rel) = clean[search..].find("<invoke name=") {
                let start = search + rel;
                match clean[start..].find("</invoke>") {
                    Some(e) => {
                        let end = start + e + "</invoke>".len();
                        clean.drain(start..end);
                        search = start;
                    }
                    None => break,
                }
            }
            let trimmed = clean.trim();
            let content = if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
            return (content, bare_invoke_calls);
        }
    }

    // Fallback 3: look for JSON tool calls in code blocks or bare JSON.
    // When the model ignores the XML format entirely and writes tool invocations
    // as JSON in markdown code blocks or bare Hermes-style JSON, catch them here.
    // This is the "guaranteed catch-all" — if the model wrote a valid tool name
    // with arguments in any recognizable JSON format, we'll find it.
    if calls.is_empty() {
        let json_calls = parse_json_fallback_calls(text);
        if !json_calls.is_empty() {
            // Strip the JSON source from content
            let mut clean_content = text.to_string();
            for pattern in extract_json_code_blocks(text) {
                clean_content = clean_content.replace(&pattern, "");
            }
            let clean = clean_content.trim().to_string();
            let content = if clean.is_empty() { None } else { Some(clean) };
            return (content, json_calls);
        }
    }

    let content = if content_parts.is_empty() {
        None
    } else {
        Some(content_parts.join("\n"))
    };
    (content, calls)
}
