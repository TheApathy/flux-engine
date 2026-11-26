// SPDX-License-Identifier: AGPL-3.0-only

//! Server-side repair for `response_format=json_object` output that the
//! BPE-merged JSON-opener token corrupts at grammar step 0.

use crate::openai::{ChatCompletionRequest, ResponseFormat};

/// Repair the malformed-prefix output that `response_format=json_object`
/// produces on Qwen3.x tokenizers: the model's first emitted token at
/// JSON-grammar step 0 is often the BPE-merged `{"` (Qwen vocab id 4754)
/// because object-with-key is the densest JSON-object opening in training.
/// xgrammar's bitmask correctly accepts `{"` as a valid 2-char JSON prefix
/// (open object, open key string), but the model continues as if writing a
/// fresh top-level object inside the now-open key string, producing
/// `{"{"score":50,"reason":"test"}` (an unterminated outer object whose key
/// is itself a JSON object literal as raw text).
///
/// Server-side band-aid: when the emitted content fails to parse as
/// JSON but parses cleanly with the opening `{"` stripped, drop that
/// spurious opener while preserving JSON-legal surrounding whitespace.
/// Non-JSON leading Unicode whitespace is removed only when the result parses
/// as JSON. The streaming path has the same bug; a grammar-level fix that
/// pre-advances the matcher past `{` is the proper long-term solution and is
/// tracked separately.
pub(crate) fn repair_json_object_prefix(req: &ChatCompletionRequest, content: String) -> String {
    if !matches!(req.response_format, Some(ResponseFormat::JsonObject)) {
        return content;
    }
    repair_inner(content)
}

/// Strip an outer markdown JSON code fence (`` ```json\n...\n``` `` or
/// `` ```\n...\n``` ``) when `response_format` requires JSON (`json_object` or
/// `json_schema`).
///
/// The joint `tools | response_format` XGrammar union allows an optional
/// markdown fence around schema-valid JSON so thinking models whose top-1
/// post-`</think>` logit is `` ``` `` do not collapse into spurious
/// `<tool_call>` loops when `` ``` `` is masked out. This unwrapper restores
/// pure JSON on the OpenAI wire whenever the fenced payload parses cleanly as
/// JSON.
pub(crate) fn strip_json_markdown_fence(req: &ChatCompletionRequest, content: String) -> String {
    if !matches!(
        req.response_format,
        Some(ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. })
    ) {
        return content;
    }
    strip_json_markdown_fence_inner(content)
}

pub(crate) fn strip_json_markdown_fence_inner(content: String) -> String {
    let trimmed = content.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\r' | '\n'));
    let Some(after_open) = trimmed.strip_prefix("```") else {
        return content;
    };
    let Some(nl_idx) = after_open.find('\n') else {
        return content;
    };
    let lang = after_open[..nl_idx].trim_matches(|c: char| matches!(c, ' ' | '\t' | '\r'));
    if !(lang.is_empty() || lang.eq_ignore_ascii_case("json")) {
        return content;
    }
    let body_and_close = &after_open[nl_idx + 1..];
    let Some(inner) = body_and_close.strip_suffix("```") else {
        return content;
    };
    let inner_trimmed = inner.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\r' | '\n'));
    if serde_json::from_str::<serde_json::Value>(inner_trimmed).is_ok() {
        inner_trimmed.to_string()
    } else {
        content
    }
}

/// Incremental streaming unwrapper for an outer `` ```json\n...\n``` `` fence
/// when `response_format` is `json_object` or `json_schema`.
///
/// Zero-latency passthrough when the first non-whitespace byte is not `` ` ``
/// (e.g. raw `{...}` JSON). When an opening fence is detected, drops the
/// opening fence line and holds only the trailing whitespace + up to 3
/// backticks so embedded `` ``` `` inside JSON string values are forwarded
/// verbatim while the terminal closing `` ``` `` before EOF is stripped.
#[derive(Debug)]
pub(crate) struct JsonFenceStripper {
    phase: FencePhase,
}

#[derive(Debug)]
enum FencePhase {
    Init { buf: String },
    InsideFence { tail: String },
    Passthrough,
}

impl JsonFenceStripper {
    pub(crate) fn new() -> Self {
        Self {
            phase: FencePhase::Init { buf: String::new() },
        }
    }

    pub(crate) fn push(&mut self, delta: &str) -> String {
        if delta.is_empty() {
            return String::new();
        }
        match &mut self.phase {
            FencePhase::Passthrough => delta.to_string(),
            FencePhase::InsideFence { tail } => {
                tail.push_str(delta);
                drain_safe_fence_body(tail)
            }
            FencePhase::Init { buf } => {
                buf.push_str(delta);
                let ws_end = buf
                    .find(|c: char| !matches!(c, ' ' | '\t' | '\r' | '\n'))
                    .unwrap_or(buf.len());
                let rest = &buf[ws_end..];
                if rest.is_empty() {
                    if buf.len() > 64 {
                        let out = std::mem::take(buf);
                        self.phase = FencePhase::Passthrough;
                        return out;
                    }
                    return String::new();
                }
                if !rest.starts_with('`') {
                    let out = std::mem::take(buf);
                    self.phase = FencePhase::Passthrough;
                    return out;
                }
                if let Some(after_ticks) = rest.strip_prefix("```") {
                    if let Some(nl_idx) = after_ticks.find('\n') {
                        let lang = after_ticks[..nl_idx]
                            .trim_matches(|c: char| matches!(c, ' ' | '\t' | '\r'));
                        if lang.is_empty() || lang.eq_ignore_ascii_case("json") {
                            let body_start = ws_end + 3 + nl_idx + 1;
                            let mut tail = buf[body_start..].to_string();
                            let out = drain_safe_fence_body(&mut tail);
                            self.phase = FencePhase::InsideFence { tail };
                            return out;
                        }
                        let out = std::mem::take(buf);
                        self.phase = FencePhase::Passthrough;
                        return out;
                    }
                    if rest.len() > 24 {
                        let out = std::mem::take(buf);
                        self.phase = FencePhase::Passthrough;
                        return out;
                    }
                    return String::new();
                }
                if "```".starts_with(rest) {
                    return String::new();
                }
                let out = std::mem::take(buf);
                self.phase = FencePhase::Passthrough;
                out
            }
        }
    }

    pub(crate) fn flush(&mut self) -> String {
        match std::mem::replace(&mut self.phase, FencePhase::Passthrough) {
            FencePhase::Passthrough => String::new(),
            FencePhase::Init { buf } => buf,
            FencePhase::InsideFence { tail } => {
                let trimmed = tail.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\r' | '\n'));
                if trimmed == "```" {
                    String::new()
                } else {
                    tail
                }
            }
        }
    }
}

/// Emit the prefix of `tail` that cannot be part of a terminal closing
/// `[ \t\r\n]* ``` [ \t\r\n]*` fence, leaving only the potential closing
/// suffix in `tail`.
fn drain_safe_fence_body(tail: &mut String) -> String {
    let bytes = tail.as_bytes();
    let mut i = bytes.len();
    // 1. Trailing ASCII whitespace after a possible closing ```.
    while i > 0 && matches!(bytes[i - 1], b' ' | b'\t' | b'\r' | b'\n') {
        i -= 1;
    }
    // 2. Up to 3 contiguous backticks forming a partial or full ``` close.
    let mut ticks = 0usize;
    while i > 0 && bytes[i - 1] == b'`' && ticks < 3 {
        i -= 1;
        ticks += 1;
    }
    // 3. Leading ASCII whitespace before the closing ``` (e.g. `\n``` `).
    while i > 0 && matches!(bytes[i - 1], b' ' | b'\t' | b'\r' | b'\n') {
        i -= 1;
    }
    if i == 0 {
        return String::new();
    }
    let remainder = tail.split_off(i);
    std::mem::replace(tail, remainder)
}

fn repair_inner(content: String) -> String {
    if serde_json::from_str::<serde_json::Value>(&content).is_ok() {
        return content;
    }
    let after_whitespace = content.trim_start();
    let prefix_len = content.len() - after_whitespace.len();
    let legal_prefix: String = content[..prefix_len]
        .chars()
        .filter(|ch| matches!(*ch, ' ' | '\t' | '\r' | '\n'))
        .collect();
    if legal_prefix.len() != prefix_len
        && serde_json::from_str::<serde_json::Value>(after_whitespace).is_ok()
    {
        let repaired = legal_prefix + after_whitespace;
        tracing::info!(
            "json_object: removed non-JSON leading whitespace from response. \
             Original len={}, repaired len={}",
            content.len(),
            repaired.len(),
        );
        return repaired;
    }
    if !after_whitespace.starts_with("{\"") {
        return content;
    }
    let stripped = &after_whitespace[2..];
    if serde_json::from_str::<serde_json::Value>(stripped).is_ok() {
        let mut repaired = String::with_capacity(legal_prefix.len() + stripped.len());
        repaired.push_str(&legal_prefix);
        repaired.push_str(stripped);
        tracing::info!(
            "json_object: stripped spurious leading `{{\\\"` from response \
             (issue #43; BPE-merged token at grammar step 0). \
             Original len={}, repaired len={}",
            content.len(),
            repaired.len(),
        );
        return repaired;
    }
    content
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_already_valid_json() {
        let s = "{\"score\":50}".to_string();
        assert_eq!(repair_inner(s.clone()), s);
    }

    #[test]
    fn strips_spurious_prefix_when_strip_yields_valid_json() {
        let bug = "{\"{\"score\":50,\"reason\":\"test\"}".to_string();
        let repaired = repair_inner(bug);
        assert_eq!(repaired, "{\"score\":50,\"reason\":\"test\"}");
        let _: serde_json::Value = serde_json::from_str(&repaired).unwrap();
    }

    #[test]
    fn passthrough_when_strip_does_not_yield_valid_json() {
        let s = "{\"unterminated".to_string();
        assert_eq!(repair_inner(s.clone()), s);
    }

    #[test]
    fn handles_trailing_whitespace() {
        let bug = "{\"{\"x\":1}\n".to_string();
        let repaired = repair_inner(bug);
        assert_eq!(repaired, "{\"x\":1}\n");
    }

    #[test]
    fn passthrough_when_does_not_start_with_quoted_brace() {
        let s = "Sorry, I cannot do that.".to_string();
        assert_eq!(repair_inner(s.clone()), s);
    }

    #[test]
    fn valid_json_preserves_json_legal_surrounding_whitespace() {
        for prefix in ["", " \t\r\n"] {
            for suffix in ["", "\r\n\t "] {
                let content = format!("{prefix}{{\"score\":50}}{suffix}");
                assert_eq!(repair_inner(content.clone()), content);
            }
        }
    }

    #[test]
    fn malformed_opener_repair_preserves_json_legal_prefix_and_suffix_bytes() {
        for prefix in ["", " \t\r\n"] {
            for suffix in ["", "\r\n\t "] {
                let payload = "{\"score\":50,\"reason\":\"test\"}";
                let malformed = format!("{prefix}{{\"{payload}{suffix}");
                let original_len = malformed.len();
                let repaired = repair_inner(malformed);
                assert_eq!(repaired, format!("{prefix}{payload}{suffix}"));
                assert_eq!(repaired.len(), original_len - 2);
                let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
                assert_eq!(parsed["score"], 50);
            }
        }
    }

    #[test]
    fn repair_preserves_whitespace_between_spurious_opener_and_object() {
        let content = "\u{2003}\t{\" \n{\"x\":1}\r\n".to_string();
        assert_eq!(repair_inner(content), "\t \n{\"x\":1}\r\n");
    }

    #[test]
    fn invalid_leading_unicode_whitespace_is_repaired_around_valid_json() {
        for (prefix, legal_prefix) in [
            ("\u{00a0}", ""),
            ("\u{2003}\u{3000}", ""),
            (" \u{00a0}\t\u{2003}\r\n", " \t\r\n"),
        ] {
            let content = format!("{prefix}{{\"x\":1}}\r\n");
            assert!(serde_json::from_str::<serde_json::Value>(&content).is_err());
            let repaired = repair_inner(content);
            assert_eq!(repaired, format!("{legal_prefix}{{\"x\":1}}\r\n"));
            let _: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        }
    }

    #[test]
    fn invalid_leading_unicode_whitespace_and_malformed_opener_are_repaired_together() {
        let content = " \u{00a0}\t\u{2003}\r\n{\"{\"x\":1}\t\n".to_string();
        let repaired = repair_inner(content);
        assert_eq!(repaired, " \t\r\n{\"x\":1}\t\n");
        let _: serde_json::Value = serde_json::from_str(&repaired).unwrap();
    }

    #[test]
    fn invalid_unicode_suffix_does_not_trigger_a_partial_repair() {
        for content in ["\u{00a0}{\"x\":1}\u{2003}", " \t{\"{\"x\":1}\u{2003}"] {
            assert!(serde_json::from_str::<serde_json::Value>(content).is_err());
            assert_eq!(repair_inner(content.to_string()), content);
        }
    }

    #[test]
    fn unrepairable_json_preserves_ascii_and_unicode_whitespace() {
        for prefix in [" \t\n", "\u{00a0}\u{2003}"] {
            for payload in ["{\"unterminated", "{\"{\"x\":}", "{\"not json"] {
                let content = format!("{prefix}{payload}\n\u{3000}");
                assert_eq!(repair_inner(content.clone()), content);
            }
        }
    }

    #[test]
    fn nonmatching_openers_and_whitespace_only_content_are_unchanged() {
        for content in [
            "",
            " \t\n",
            "\u{00a0}\u{2003}",
            "\u{2003}Sorry, I cannot do that.\n",
            "\t[1,2]\n",
            "prefix {\"{\"x\":1}",
        ] {
            assert_eq!(repair_inner(content.to_string()), content);
        }
    }

    fn request_with_format(response_format: Option<serde_json::Value>) -> ChatCompletionRequest {
        let mut request = serde_json::json!({
            "model": "repair-json-test",
            "messages": [{"role": "user", "content": "Return JSON."}]
        });
        if let Some(response_format) = response_format {
            request["response_format"] = response_format;
        }
        serde_json::from_value(request).unwrap()
    }

    #[test]
    fn request_level_repair_is_gated_to_json_object_format() {
        let content = "\u{2003}\t{\"{\"x\":1}\n".to_string();
        for response_format in [
            None,
            Some(serde_json::json!({"type": "text"})),
            Some(serde_json::json!({
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "object"}}
            })),
        ] {
            let request = request_with_format(response_format);
            assert_eq!(
                repair_json_object_prefix(&request, content.clone()),
                content
            );
        }
        let request = request_with_format(Some(serde_json::json!({"type": "json_object"})));
        assert_eq!(
            repair_json_object_prefix(&request, content),
            "\t{\"x\":1}\n"
        );
    }

    #[test]
    fn strip_json_markdown_fence_unwraps_only_valid_json_fences_for_json_formats() {
        let fenced = "\n\n```json\n{\n  \"date\": \"2026-03-20\",\n  \"items\": [1, 2]\n}\n```\n";
        let expected = "{\n  \"date\": \"2026-03-20\",\n  \"items\": [1, 2]\n}";
        let req_schema = request_with_format(Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "briefing", "schema": {"type": "object"}}
        })));
        assert_eq!(
            strip_json_markdown_fence(&req_schema, fenced.to_string()),
            expected
        );
        let req_obj = request_with_format(Some(serde_json::json!({"type": "json_object"})));
        assert_eq!(
            strip_json_markdown_fence(&req_obj, "```\n{\"ok\":true}\n```".to_string()),
            "{\"ok\":true}"
        );
        // Untouched when response_format is None or Text.
        let req_none = request_with_format(None);
        assert_eq!(
            strip_json_markdown_fence(&req_none, fenced.to_string()),
            fenced
        );
        // Untouched for non-JSON language tag.
        let python_fence = "```python\n{\"ok\":true}\n```";
        assert_eq!(
            strip_json_markdown_fence(&req_schema, python_fence.to_string()),
            python_fence
        );
    }

    #[test]
    fn streaming_fence_stripper_handles_token_chunks_and_embedded_backticks() {
        let payload = "{\n  \"code\": \"hello\\n```python\\nx = 1\\n```\",\n  \"ok\": true\n}";
        let full = format!("\n\n```json\n{payload}\n```\n");

        // Test byte-by-byte and realistic multi-token chunking.
        for chunk_size in [1usize, 2, 3, 5, 11, full.len()] {
            let mut stripper = JsonFenceStripper::new();
            let mut out = String::new();
            let bytes = full.as_bytes();
            let mut pos = 0;
            while pos < bytes.len() {
                let end = (pos + chunk_size).min(bytes.len());
                out.push_str(&stripper.push(&full[pos..end]));
                pos = end;
            }
            out.push_str(&stripper.flush());
            assert_eq!(out, payload, "chunk_size={chunk_size}");
            let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
            assert_eq!(parsed["ok"], true);
        }

        // Raw JSON passes through immediately on the first non-whitespace byte.
        let mut raw_stripper = JsonFenceStripper::new();
        assert_eq!(raw_stripper.push("\n\n"), "");
        assert_eq!(raw_stripper.push("{"), "\n\n{");
        assert_eq!(raw_stripper.push("\"x\":1}"), "\"x\":1}");
        assert_eq!(raw_stripper.flush(), "");
    }
}
