// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Outer tag family currently open in [`StreamingToolDetector`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EnvelopeKind {
    /// `<tool_call>...</tool_call>` (Qwen3-Coder, Hermes, GLM native).
    QwenOrHermes,
    /// `<|tool_call>...<tool_call|>` (Gemma 4).
    Gemma4,
    /// `<minimax:tool_call>...</minimax:tool_call>` (MiniMax canonical).
    MinimaxCanonical,
    /// `<minimax:_call>...</minimax:_call>` (MiniMax BPE-broken — F73 / fix42).
    MinimaxBroken,
}

impl EnvelopeKind {
    pub(super) const fn open_tag(self) -> &'static str {
        match self {
            Self::QwenOrHermes => "<tool_call>",
            Self::Gemma4 => "<|tool_call>",
            Self::MinimaxCanonical => "<minimax:tool_call>",
            Self::MinimaxBroken => "<minimax:_call>",
        }
    }

    /// Locate this envelope's matching closing tag in `buffer`, returning
    /// `(byte_offset, close_tag_len)`.
    ///
    /// For `<tool_call>` envelopes this delegates to `glm_xml_scan::legacy_close`
    /// so literal `</tool_call>` substrings inside `<parameter=...>...</parameter>`
    /// values never truncate the tool call.
    pub(super) fn find_close(self, buffer: &str) -> Option<(usize, usize)> {
        match self {
            Self::QwenOrHermes => {
                glm_xml_scan::legacy_close(buffer).map(|pos| (pos, glm_xml_scan::CLOSE.len()))
            }
            Self::Gemma4 => buffer
                .find("<tool_call|>")
                .map(|pos| (pos, "<tool_call|>".len())),
            Self::MinimaxCanonical => buffer
                .find("</minimax:tool_call>")
                .map(|pos| (pos, "</minimax:tool_call>".len())),
            // Live MiniMax M2.7 BPE-broken `<minimax:_call>` openers can pair
            // with either `</minimax:_call>` or canonical `</minimax:tool_call>`,
            // but must never close on foreign `</tool_call>` or `<tool_call|>`.
            Self::MinimaxBroken => ["</minimax:_call>", "</minimax:tool_call>"]
                .into_iter()
                .filter_map(|tag| buffer.find(tag).map(|pos| (pos, tag.len())))
                .min_by_key(|(pos, _)| *pos),
        }
    }
}

/// Buffers streaming text and detects tool-call envelopes in real-time.
/// Emits structured tool-call events as envelopes arrive:
/// - `ToolCallStart` early as soon as the function name is extracted
/// - `ToolCallDelta` once when the envelope closes and arguments are serialized
/// - `ToolCallEnd` immediately after `ToolCallDelta` when the close tag is found
pub struct StreamingToolDetector {
    pub(super) buffer: String,
    /// Active outer envelope currently being buffered (`None` outside tags).
    pub(super) active_envelope: Option<EnvelopeKind>,
    /// Configured parser whose canonical wire grammar is expected. `None`
    /// preserves the legacy detector API for callers that do not consume
    /// parser provenance.
    pub(super) expected_parser: Option<String>,
    /// Number of non-canonical parser recoveries since the last drain.
    pub(super) recovery_count: usize,
    /// Exact post-thinking wire transcript retained until terminal flush for
    /// side-effect-free mixed-format auditing. Only populated for Qwen.
    pub(super) qwen_audit_transcript: String,
    /// Recovery was already observed for this content phase. This survives a
    /// count drain so terminal residual auditing cannot duplicate an event.
    pub(super) qwen_recovery_observed: bool,
    pub(super) call_counter: u32,
    /// Track if any tool calls were emitted during process() to prevent
    /// flush() from re-emitting them (causes duplicate arguments in stream).
    pub(super) emitted_tool_calls: bool,
    /// For early header streaming: name already emitted for current tool call.
    pub(super) current_tc_name: Option<String>,
    /// For early header streaming: ID of the current in-progress tool call.
    pub(super) current_tc_id: Option<String>,
}

pub enum DetectorOutput {
    /// Plain text content (not a tool call).
    Content(String),
    /// Complete tool call (used by flush/blocking path).
    ToolCall(ToolCall, usize),
    /// Tool call header (name + id). Emitted early once the function name is known.
    ToolCallStart {
        id: String,
        name: String,
        idx: usize,
    },
    /// Serialized JSON arguments, emitted once when the tool-call envelope closes.
    ToolCallDelta { args: String, idx: usize },
    /// Tool call complete (closing tag seen).
    ToolCallEnd { idx: usize },
}

fn sanitize_xml_func_name(raw: &str) -> Option<String> {
    let mut name = raw.trim().to_string();
    if let Some(rest) = name.strip_prefix("name")
        && let Some(after_eq) = rest.trim_start().strip_prefix('=')
    {
        name = after_eq
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
    }
    name = name.trim_matches('"').trim_matches('\'').to_string();
    // Sanitize: model may generate "Bash=bashash" or "Bash=Bash" at long context.
    if let Some(eq_pos) = name.find('=') {
        name = name[..eq_pos].trim().to_string();
    }
    (!name.is_empty()).then_some(name)
}

fn extract_gemma4_prefix_name(trimmed: &str) -> Option<String> {
    let after = trimmed
        .strip_prefix("call:")
        .or_else(|| trimmed.strip_prefix("_call:"))?;
    let end = after.find('{')?;
    let name = after[..end].trim();
    (!name.is_empty() && !name.contains(char::is_whitespace)).then(|| name.to_string())
}

/// Skip a JSON string starting at `bytes[start] == b'"'`, returning the byte
/// offset immediately after its closing `"` (or `None` if unclosed).
fn skip_json_string(bytes: &[u8], start: usize) -> Option<usize> {
    if bytes.get(start) != Some(&b'"') {
        return None;
    }
    let mut i = start + 1;
    let mut escaped = false;
    while i < bytes.len() {
        let b = bytes[i];
        if escaped {
            escaped = false;
        } else if b == b'\\' {
            escaped = true;
        } else if b == b'"' {
            return Some(i + 1);
        }
        i += 1;
    }
    None
}

/// Skip a JSON value starting at `bytes[start]` inside a top-level object,
/// returning the byte offset immediately after it (or `None` if incomplete).
fn skip_json_value(bytes: &[u8], start: usize) -> Option<usize> {
    let first = *bytes.get(start)?;
    match first {
        b'"' => skip_json_string(bytes, start),
        b'{' | b'[' => {
            let mut depth: i32 = 0;
            let mut i = start;
            let mut in_string = false;
            let mut escaped = false;
            while i < bytes.len() {
                let b = bytes[i];
                if in_string {
                    if escaped {
                        escaped = false;
                    } else if b == b'\\' {
                        escaped = true;
                    } else if b == b'"' {
                        in_string = false;
                    }
                } else {
                    match b {
                        b'"' => in_string = true,
                        b'{' | b'[' => depth += 1,
                        b'}' | b']' => {
                            depth -= 1;
                            if depth == 0 {
                                return Some(i + 1);
                            }
                        }
                        _ => {}
                    }
                }
                i += 1;
            }
            None
        }
        _ => {
            let mut i = start;
            while i < bytes.len() {
                let b = bytes[i];
                if b.is_ascii_whitespace() || b == b',' || b == b'}' {
                    return (i > start).then_some(i);
                }
                i += 1;
            }
            None
        }
    }
}

/// Extract the top-level `"name"` string field from a (possibly partial) JSON
/// object starting with `{`. Never descends into nested objects such as
/// `"arguments": {"name": "Alice"}`.
fn extract_top_level_json_name(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut i = 1;
    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'"' {
            return None;
        }
        let key_end = skip_json_string(bytes, i)?;
        let key: String = serde_json::from_str(&s[i..key_end]).ok()?;
        i = key_end;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if bytes.get(i) != Some(&b':') {
            return None;
        }
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if key == "name" {
            let val_end = skip_json_string(bytes, i)?;
            let val: String = serde_json::from_str(&s[i..val_end]).ok()?;
            return (!val.is_empty()).then_some(val);
        }
        i = skip_json_value(bytes, i)?;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if bytes.get(i) == Some(&b',') {
            i += 1;
        } else {
            return None;
        }
    }
}

/// Extract function name from the trimmed prefix of a partial tool-call envelope.
///
/// Anchored strictly to the active [`EnvelopeKind`] and the start of the
/// envelope body so argument content (e.g. `"name": "Alice"`, `call:foo{`, or
/// `[TOOL_CALLS]` inside parameter values) can never spoof the staged tool name.
pub(super) fn extract_streaming_name(buffer: &str, envelope: EnvelopeKind) -> Option<String> {
    let trimmed = buffer.trim_start();
    match envelope {
        EnvelopeKind::MinimaxCanonical | EnvelopeKind::MinimaxBroken => {
            // MiniMax envelopes can hold multiple `<invoke>` blocks and emit
            // complete `ToolCall`s at close; never scan inside parameter bodies.
            None
        }
        EnvelopeKind::Gemma4 => extract_gemma4_prefix_name(trimmed),
        EnvelopeKind::QwenOrHermes => {
            // Qwen3-Coder XML attribute-style: `<function=NAME>`, `<|function=NAME>`,
            // `<function NAME>`, `<function name="NAME">`, `<|function NAME>`.
            if let Some(after) = trimmed
                .strip_prefix("<function=")
                .or_else(|| trimmed.strip_prefix("<|function="))
                .or_else(|| trimmed.strip_prefix("<function "))
                .or_else(|| trimmed.strip_prefix("<|function "))
            {
                let end = after.find(['>', '\n', '<'])?;
                return sanitize_xml_func_name(&after[..end]);
            }
            // Tag-style XML: `<function>NAME</function>`.
            if let Some(after) = trimmed.strip_prefix("<function>") {
                let end = after.find("</function>")?;
                let raw = after[..end].trim();
                return (!raw.is_empty() && !raw.contains('<')).then(|| raw.to_string());
            }
            // Gemma-4 `call:NAME{` / `_call:NAME{` at the start of `<tool_call>`.
            if trimmed.starts_with("call:") || trimmed.starts_with("_call:") {
                return extract_gemma4_prefix_name(trimmed);
            }
            // Hermes JSON: top-level `"name":"X"` only (never inside `"arguments"`).
            if trimmed.starts_with('{') {
                return extract_top_level_json_name(trimmed);
            }
            None
        }
    }
}

impl Default for StreamingToolDetector {
    fn default() -> Self {
        Self::new()
    }
}
