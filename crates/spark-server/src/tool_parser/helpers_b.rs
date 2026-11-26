// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Parse all Mistral native tool calls from a completed response text.
/// Returns `(content_before_first_call, tool_calls)`.
pub(super) fn parse_mistral_native_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut content: Option<String> = None;
    let first_tag = match text.find(MISTRAL_TOOL_CALLS_TAG) {
        Some(p) => p,
        None => return (None, calls),
    };
    let before = text[..first_tag].trim();
    if !before.is_empty() {
        content = Some(before.to_string());
    }
    // Split on the tag; first element is empty (everything before the first tag
    // was already captured as content). Remaining elements are each call's
    // `name[ARGS]{json}` segment.
    let segments = text[first_tag..].split(MISTRAL_TOOL_CALLS_TAG).skip(1);
    for segment in segments {
        if segment.trim().is_empty() {
            continue;
        }
        if let Some(tc) = parse_mistral_native_call(segment) {
            calls.push(tc);
        }
    }
    (content, calls)
}

/// Format a JSON value in Gemma-4 native notation.
pub(super) fn format_gemma4_value(out: &mut String, val: &serde_json::Value) {
    match val {
        serde_json::Value::String(s) => {
            out.push_str("<|\"|>");
            out.push_str(s);
            out.push_str("<|\"|>");
        }
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => out.push_str(&n.to_string()),
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                format_gemma4_value(out, item);
            }
            out.push(']');
        }
        serde_json::Value::Object(obj) => {
            out.push('{');
            let mut first = true;
            for (key, v) in obj {
                if !first {
                    out.push(',');
                }
                first = false;
                out.push_str(key);
                out.push(':');
                format_gemma4_value(out, v);
            }
            out.push('}');
        }
    }
}

/// Parse a Gemma-4 native tool call: `call:fn_name{key:val,...}`
/// Converts native `<|"|>` delimited values to JSON.
pub(super) fn parse_gemma4_native_call(text: &str) -> Option<ToolCall> {
    let text = text.trim();
    // Accept both `call:` and `_call:` (sentencepiece `▁call:` → `_call:` in some tokenizer decodes)
    let rest = text
        .strip_prefix("call:")
        .or_else(|| text.strip_prefix("_call:"))?;
    let brace_pos = rest.find('{')?;
    let name = rest[..brace_pos].trim().to_string();
    if name.is_empty() {
        return None;
    }

    // Extract the arguments between `{` and the last `}` (dropping any trailing text).
    let args_str = match rest[brace_pos..].rfind('}') {
        Some(end_rel) => &rest[brace_pos..=brace_pos + end_rel],
        None => &rest[brace_pos..],
    };
    // Convert native format to JSON:
    // 1. Replace <|"|> with "
    // 2. Quote unquoted keys
    let json_str = gemma4_to_json(args_str);
    let arguments = if serde_json::from_str::<serde_json::Value>(&json_str).is_ok() {
        json_str
    } else if let Some(end) = find_balanced_json_end(&json_str)
        && serde_json::from_str::<serde_json::Value>(&json_str[..end]).is_ok()
    {
        json_str[..end].to_string()
    } else {
        // Fallback: attempt bare-object JSON repair before giving up.
        let repaired = repair_bare_object_json(&json_str);
        if serde_json::from_str::<serde_json::Value>(&repaired).is_ok() {
            repaired
        } else {
            "{}".to_string()
        }
    };

    Some(ToolCall {
        id: next_tool_call_id(),
        call_type: "function".into(),
        function: FunctionCall { name, arguments },
    })
}

/// Append `raw` as a JSON-escaped double-quoted string literal (RFC 8259).
fn push_json_string(out: &mut String, raw: &str) {
    use std::fmt::Write as _;

    out.push('"');
    for ch in raw.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Convert Gemma-4 native `{key:<|"|>val<|"|>,...}` to JSON `{"key":"val",...}`.
pub(super) fn gemma4_to_json(native: &str) -> String {
    use std::fmt::Write as _;

    const GEMMA_QUOTE: &str = "<|\"|>";
    // Convert `<|"|>...<|"|>` strings directly with proper JSON escaping
    // (preserving embedded `"`, `\`, `\n`, `\r`, `\t`, and control chars),
    // and quote unquoted keys (`key:` → `"key":`) while preserving regular
    // quoted strings, JSON booleans/null, and JSON numbers (`10`, `0.7`, `-5`, `1e-5`).
    let mut result = String::with_capacity(native.len() + 32);
    let mut i = 0;
    while let Some(ch) = native[i..].chars().next() {
        if native[i..].starts_with(GEMMA_QUOTE) {
            let content_start = i + GEMMA_QUOTE.len();
            if let Some(rel_end) = native[content_start..].find(GEMMA_QUOTE) {
                push_json_string(&mut result, &native[content_start..content_start + rel_end]);
                i = content_start + rel_end + GEMMA_QUOTE.len();
            } else {
                push_json_string(&mut result, &native[content_start..]);
                i = native.len();
            }
        } else if ch == '"' {
            // Preserve regular double-quoted JSON strings (escaping raw control chars).
            result.push('"');
            i += 1;
            while let Some(c) = native[i..].chars().next() {
                if c == '"' {
                    result.push('"');
                    i += 1;
                    break;
                }
                if c == '\\' {
                    result.push('\\');
                    i += 1;
                    if let Some(esc) = native[i..].chars().next() {
                        result.push(esc);
                        i += esc.len_utf8();
                    }
                    continue;
                }
                match c {
                    '\n' => result.push_str("\\n"),
                    '\r' => result.push_str("\\r"),
                    '\t' => result.push_str("\\t"),
                    '\u{08}' => result.push_str("\\b"),
                    '\u{0C}' => result.push_str("\\f"),
                    ctrl if (ctrl as u32) < 0x20 => {
                        let _ = write!(result, "\\u{:04x}", ctrl as u32);
                    }
                    other => result.push(other),
                }
                i += c.len_utf8();
            }
        } else if (ch == '-'
            && native[i + 1..]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit()))
            || ch.is_ascii_digit()
        {
            // Scan a JSON number literal; only emit unquoted if it is a complete
            // number token (not a key followed by `:` or an identifier like `123abc`).
            let start = i;
            if ch == '-' {
                i += 1;
            }
            while let Some(c) = native[i..].chars().next() {
                if c.is_ascii_digit() {
                    i += 1;
                } else {
                    break;
                }
            }
            if native[i..].starts_with('.')
                && native[i + 1..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_digit())
            {
                i += 1;
                while let Some(c) = native[i..].chars().next() {
                    if c.is_ascii_digit() {
                        i += 1;
                    } else {
                        break;
                    }
                }
            }
            if native[i..].starts_with(['e', 'E']) {
                let exp_start = i;
                i += 1;
                if native[i..].starts_with(['+', '-']) {
                    i += 1;
                }
                let digits_start = i;
                while let Some(c) = native[i..].chars().next() {
                    if c.is_ascii_digit() {
                        i += 1;
                    } else {
                        break;
                    }
                }
                if i == digits_start {
                    i = exp_start;
                }
            }
            let next_ch = native[i..].chars().next();
            if next_ch.is_some_and(|c| c.is_alphanumeric() || c == '_') {
                // Actually an alphanumeric identifier starting with digits (e.g. `123abc`).
                while let Some(c) = native[i..].chars().next() {
                    if c.is_alphanumeric() || c == '_' {
                        i += c.len_utf8();
                    } else {
                        break;
                    }
                }
                result.push('"');
                result.push_str(&native[start..i]);
                result.push('"');
            } else if next_ch == Some(':') {
                result.push('"');
                result.push_str(&native[start..i]);
                result.push('"');
            } else {
                result.push_str(&native[start..i]);
            }
        } else if ch.is_alphanumeric() || ch == '_' {
            // Potential unquoted key — accumulate until we see `:` or non-key char
            let start = i;
            while let Some(c) = native[i..].chars().next() {
                if c.is_alphanumeric() || c == '_' {
                    i += c.len_utf8();
                } else {
                    break;
                }
            }
            let word = &native[start..i];
            if native[i..].starts_with(':') {
                // This is a key — quote it
                result.push('"');
                result.push_str(word);
                result.push('"');
            } else {
                // Not a key — check if it's a keyword (true/false/null)
                match word {
                    "true" | "false" | "null" => result.push_str(word),
                    _ => {
                        // Unknown bare word — quote it as a string
                        result.push('"');
                        result.push_str(word);
                        result.push('"');
                    }
                }
            }
        } else {
            result.push(ch);
            i += ch.len_utf8();
        }
    }
    result
}

/// Shared tool_choice instruction appended to parser system prompts.
///
/// `format_desc` names the parser's wire format (e.g. `"a <tool_call> block"`,
/// `"a bare JSON tool-call object"`, `"a <minimax:tool_call> block"`, or
/// `"a [TOOL_CALLS] invocation"`).
pub(super) fn append_tool_choice_instruction(
    prompt: &mut String,
    tool_choice: &ToolChoice,
    format_desc: &str,
) {
    use std::fmt::Write as _;

    match tool_choice {
        ToolChoice::Mode(s) if s == "required" => {
            let _ = write!(
                prompt,
                "\n\n<IMPORTANT>\nYou MUST call at least one function. \
                 Do NOT respond with text — respond ONLY with {format_desc}.\n</IMPORTANT>",
            );
        }
        ToolChoice::Specific { function } => {
            let _ = write!(
                prompt,
                "\n\n<IMPORTANT>\nYou MUST call the '{}' function. \
                 Do NOT respond with text — respond ONLY with {format_desc} \
                 calling '{}'.\n</IMPORTANT>",
                function.name, function.name,
            );
        }
        _ => {}
    }
}

/// Render the `<tools>` body for a parser's `system_prompt()`.
///
/// When TSCG is enabled (MODEL.toml `[behavior].tscg`), returns the
/// compact function-signature block. Otherwise calls `render_json` —
/// the parser's existing JSON serialization — so the TSCG-off path is
/// byte-identical to before this helper existed.
pub(super) fn tool_list_body(
    tools: &[ToolDefinition],
    render_json: impl FnOnce() -> String,
) -> String {
    if crate::tscg::tscg_enabled() {
        crate::tscg::compile_tools(tools)
    } else {
        render_json()
    }
}

// ── Output parsing (format-agnostic) ──
