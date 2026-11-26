// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

impl StreamingToolDetector {
    pub fn new() -> Self {
        Self::new_inner(None)
    }

    /// Construct a detector that also reports recoveries relative to one
    /// configured parser's canonical output grammar.
    pub fn new_with_parser(expected_parser: &str) -> Self {
        Self::new_inner(Some(expected_parser.to_string()))
    }

    fn new_inner(expected_parser: Option<String>) -> Self {
        Self {
            buffer: String::new(),
            active_envelope: None,
            expected_parser,
            recovery_count: 0,
            qwen_audit_transcript: String::new(),
            qwen_recovery_observed: false,
            call_counter: 0,
            emitted_tool_calls: false,
            current_tc_name: None,
            current_tc_id: None,
        }
    }

    /// Reset the detector state. Called when thinking→content transition occurs
    /// to prevent thinking-era tag fragments from corrupting tool detection.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.active_envelope = None;
        self.qwen_audit_transcript.clear();
        self.qwen_recovery_observed = false;
        self.current_tc_name = None;
        self.current_tc_id = None;
    }

    /// Drain parser-recovery provenance accumulated by `process`/`flush`.
    /// Callers can translate each count into their own audit or intervention
    /// representation without coupling the detector to an API response type.
    pub fn take_recovery_count(&mut self) -> usize {
        std::mem::take(&mut self.recovery_count)
    }

    pub(super) fn expects_qwen(&self) -> bool {
        self.expected_parser.as_deref() == Some("qwen3_coder")
    }

    pub(super) fn record_parser_recovery(&mut self) {
        if self.expects_qwen() {
            self.qwen_recovery_observed = true;
            self.recovery_count = self.recovery_count.saturating_add(1);
        }
    }

    /// Feed a text delta. Returns events to emit (content or tool calls).
    /// Emits `ToolCallStart` early once the function name is known, then
    /// buffers arguments until the closing tag and emits `ToolCallDelta` +
    /// `ToolCallEnd` once the full envelope parses.
    pub fn process(&mut self, new_text: &str) -> Vec<DetectorOutput> {
        let mut outputs = Vec::new();
        if self.expects_qwen() {
            self.qwen_audit_transcript.push_str(new_text);
        }
        self.buffer.push_str(new_text);
        loop {
            match self.process_glm_native(&mut outputs) {
                glm_xml_stream::NativeStep::Consumed => continue,
                glm_xml_stream::NativeStep::Pending => break,
                glm_xml_stream::NativeStep::NotNative => {}
            }
            if let Some(env) = self.active_envelope {
                // Scan ONLY for the active envelope's matching closing tag.
                // For `<tool_call>`, `EnvelopeKind::find_close` delegates to
                // `glm_xml_scan::legacy_close` so literal `</tool_call>` inside
                // `<parameter=code>...</parameter>` never truncates the call.
                if let Some((end, close_len)) = env.find_close(&self.buffer) {
                    let canonical_qwen_envelope = env == EnvelopeKind::QwenOrHermes;
                    let inner = self.buffer[..end].to_string();
                    let qwen_recovery = self.expects_qwen()
                        && (!canonical_qwen_envelope || !is_canonical_qwen_body(&inner));
                    self.buffer = self.buffer[end + close_len..].to_string();
                    self.active_envelope = None;

                    let parsed_calls = parse_envelope_calls(inner.trim(), self.call_counter);
                    let parsed_any = self.emit_envelope_calls(parsed_calls, &mut outputs);
                    if qwen_recovery || (self.expects_qwen() && !parsed_any) {
                        self.record_parser_recovery();
                    }
                    // Reset incremental state for next tool call
                    self.reset_call_state();
                    continue;
                }

                // No closing tag yet — try to extract function name for early header emission.
                // Arguments are NOT streamed incrementally because Qwen3-Coder XML format
                // (`<parameter=key>value</parameter>`) must be converted to JSON before
                // emission. The name header is emitted immediately so clients get instant
                // feedback that a tool call started.
                if self.current_tc_name.is_none()
                    && let Some(name) = extract_streaming_name(&self.buffer, env)
                {
                    let id = next_tool_call_id();
                    let idx = self.call_counter as usize;
                    outputs.push(DetectorOutput::ToolCallStart {
                        id: id.clone(),
                        name: name.clone(),
                        idx,
                    });
                    self.current_tc_name = Some(name);
                    self.current_tc_id = Some(id);
                }
                break; // Wait for more tokens (args buffered until matching close tag)
            } else {
                let earliest_envelope = [
                    EnvelopeKind::QwenOrHermes,
                    EnvelopeKind::Gemma4,
                    EnvelopeKind::MinimaxCanonical,
                    EnvelopeKind::MinimaxBroken,
                ]
                .into_iter()
                .filter_map(|env| self.buffer.find(env.open_tag()).map(|pos| (pos, env)))
                .min_by_key(|(pos, _)| *pos);

                if let Some(mistral_start) = self.buffer.find(MISTRAL_TOOL_CALLS_TAG)
                    && earliest_envelope.is_none_or(|(env_pos, _)| mistral_start < env_pos)
                {
                    // Mistral native: [TOOL_CALLS]name[ARGS]{json}
                    // No wrapping tag — emit content before the tag, then try to
                    // parse a complete segment when both [ARGS] and a balanced
                    // JSON object are present. If not yet complete, break and
                    // wait for more tokens.
                    if mistral_start > 0 {
                        let before = self.buffer[..mistral_start].to_string();
                        outputs.push(DetectorOutput::Content(before));
                        self.buffer = self.buffer[mistral_start..].to_string();
                    }
                    // Must have [ARGS] before we can extract a name.
                    let after_tag = &self.buffer[MISTRAL_TOOL_CALLS_TAG.len()..];
                    let args_rel = match after_tag.find(MISTRAL_ARGS_TAG) {
                        Some(p) => p,
                        None => break, // wait for more tokens
                    };
                    let name = after_tag[..args_rel].trim().to_string();
                    let json_abs_start =
                        MISTRAL_TOOL_CALLS_TAG.len() + args_rel + MISTRAL_ARGS_TAG.len();
                    // Skip leading whitespace before the JSON object.
                    let mut json_rel = json_abs_start;
                    while json_rel < self.buffer.len()
                        && self.buffer.as_bytes()[json_rel].is_ascii_whitespace()
                    {
                        json_rel += 1;
                    }
                    if json_rel >= self.buffer.len() || self.buffer.as_bytes()[json_rel] != b'{' {
                        break; // wait for {
                    }
                    // Look for a balanced JSON object; if not complete, break.
                    let json_tail = &self.buffer[json_rel..];
                    let Some(json_end_rel) = find_balanced_json_end(json_tail) else {
                        break; // wait for more tokens to close the JSON
                    };
                    // Emit ToolCallStart now (name is known).
                    let id = next_tool_call_id();
                    let idx = self.call_counter as usize;
                    if !name.is_empty() {
                        outputs.push(DetectorOutput::ToolCallStart {
                            id: id.clone(),
                            name: name.clone(),
                            idx,
                        });
                    }
                    // Extract and canonicalize the JSON arguments, then emit delta + end.
                    let raw_args = &json_tail[..json_end_rel];
                    let canonical = serde_json::from_str::<serde_json::Value>(raw_args)
                        .ok()
                        .and_then(|v| serde_json::to_string(&v).ok())
                        .unwrap_or_else(|| "{}".to_string());
                    // A Mistral envelope in a Qwen-configured stream is a
                    // cross-format recovery whether it publishes or is dropped.
                    self.record_parser_recovery();
                    if !name.is_empty() {
                        outputs.push(DetectorOutput::ToolCallDelta {
                            args: canonical,
                            idx,
                        });
                        outputs.push(DetectorOutput::ToolCallEnd { idx });
                        self.call_counter += 1;
                        self.emitted_tool_calls = true;
                    }
                    // Advance the buffer past the parsed JSON.
                    let consumed = json_rel + json_end_rel;
                    self.buffer = self.buffer[consumed..].to_string();
                    continue;
                } else if let Some((start, env)) = earliest_envelope {
                    // Recognised opener forms:
                    //   - `<tool_call>` (hermes / qwen3-coder, 11 chars)
                    //   - `<|tool_call>` (gemma-4, 12 chars)
                    //   - `<minimax:tool_call>` (MiniMax canonical, 19 chars)
                    //   - `<minimax:_call>` (MiniMax BPE-broken — F73 / fix42)
                    let before = self.buffer[..start].to_string();
                    self.buffer = self.buffer[start + env.open_tag().len()..].to_string();
                    self.active_envelope = Some(env);
                    if !before.is_empty() {
                        outputs.push(DetectorOutput::Content(before));
                    }
                    continue;
                } else if let Some(func_pos) = find_bare_function_opener(&self.buffer) {
                    // Bare `<function=`, `<function>`, or `<function name=` without
                    // `<tool_call>` wrapper (ignores harmless prose such as Python's
                    // `<function foo at 0x...>`).
                    if func_pos > 0 {
                        let before = self.buffer[..func_pos].to_string();
                        self.buffer = self.buffer[func_pos..].to_string();
                        outputs.push(DetectorOutput::Content(before));
                    }
                    // Check if we have a complete bare function block
                    if let Some(end) = bare_function_end(&self.buffer) {
                        let block = self.buffer[..end].to_string();
                        self.buffer = self.buffer[end..].to_string();
                        let (_, calls) = parse_bare_function_calls(&block);
                        // Wrapper-less function syntax is recovery for Qwen. This
                        // also records the branch when it consumes an unparseable
                        // block and would otherwise drop it silently.
                        self.record_parser_recovery();
                        for tc in calls {
                            let idx = self.call_counter as usize;
                            self.call_counter += 1;
                            self.emitted_tool_calls = true;
                            outputs.push(DetectorOutput::ToolCall(tc, idx));
                        }
                        continue;
                    }
                    break; // Keep buffering until function block is complete
                } else {
                    let safe = self.safe_emit_len();
                    if safe > 0 {
                        let content = self.buffer[..safe].to_string();
                        self.buffer = self.buffer[safe..].to_string();
                        if !content.is_empty() {
                            outputs.push(DetectorOutput::Content(content));
                        }
                    }
                    break;
                }
            }
        }
        outputs
    }

    /// Emit parsed calls for a completed or flushed envelope.
    ///
    /// When a speculative `ToolCallStart` was already emitted for the first
    /// call, emits an authoritative `ToolCallStart` override (preserving `id`
    /// and `idx`) if the parser resolved a different name, followed by
    /// `ToolCallDelta + ToolCallEnd`. Additional sibling calls in the same
    /// envelope emit as complete `ToolCall`s. If a header was staged but
    /// parsing yielded no calls, emits `ToolCallEnd { idx }` so downstream
    /// `tool_handlers` drops the stale unpublished staging entry.
    fn emit_envelope_calls(
        &mut self,
        parsed_calls: Vec<ToolCall>,
        outputs: &mut Vec<DetectorOutput>,
    ) -> bool {
        let idx = self.call_counter as usize;
        if let Some(staged_name) = self.current_tc_name.take() {
            let staged_id = self.current_tc_id.take();
            let mut iter = parsed_calls.into_iter();
            if let Some(first) = iter.next() {
                if first.function.name != staged_name {
                    outputs.push(DetectorOutput::ToolCallStart {
                        id: staged_id.unwrap_or(first.id),
                        name: first.function.name,
                        idx,
                    });
                }
                // Always emit when the parser produced a named call,
                // even if arguments are `{}`. Argument-less tools
                // (e.g. get_current_time) are legitimate.
                outputs.push(DetectorOutput::ToolCallDelta {
                    args: first.function.arguments,
                    idx,
                });
                outputs.push(DetectorOutput::ToolCallEnd { idx });
                self.call_counter += 1;
                self.emitted_tool_calls = true;
                for tc in iter {
                    let call_idx = self.call_counter as usize;
                    self.call_counter += 1;
                    self.emitted_tool_calls = true;
                    outputs.push(DetectorOutput::ToolCall(tc, call_idx));
                }
                true
            } else {
                tracing::warn!("Failed to parse tool call body, dropping");
                outputs.push(DetectorOutput::ToolCallEnd { idx });
                false
            }
        } else if parsed_calls.is_empty() {
            false
        } else {
            for tc in parsed_calls {
                let call_idx = self.call_counter as usize;
                self.call_counter += 1;
                self.emitted_tool_calls = true;
                outputs.push(DetectorOutput::ToolCall(tc, call_idx));
            }
            true
        }
    }

    /// Flush remaining buffer (call at stream end).
    /// Also attempts bare `<function>` detection as a last resort.
    pub fn flush(&mut self) -> Vec<DetectorOutput> {
        let outputs = self.flush_buffered();
        if self.expects_qwen()
            && !self.qwen_recovery_observed
            && qwen_residual_requires_recovery(&self.qwen_audit_transcript)
        {
            self.record_parser_recovery();
        }
        self.qwen_audit_transcript.clear();
        outputs
    }

    fn flush_buffered(&mut self) -> Vec<DetectorOutput> {
        if self.buffer.is_empty() {
            let mut out = Vec::new();
            if self.active_envelope.take().is_some() {
                self.record_parser_recovery();
                if self.current_tc_name.is_some() {
                    out.push(DetectorOutput::ToolCallEnd {
                        idx: self.call_counter as usize,
                    });
                }
                self.reset_call_state();
            }
            return out;
        }
        let text = std::mem::take(&mut self.buffer);
        let was_envelope = self.active_envelope.take();
        let canonical_qwen_envelope = was_envelope == Some(EnvelopeKind::QwenOrHermes);
        if canonical_qwen_envelope && glm_xml_scan::native_prefix(&text) {
            // Never salvage an unclosed native GLM envelope.
            self.record_parser_recovery();
            self.reset_call_state();
            return vec![DetectorOutput::Content(format!("<tool_call>{text}"))];
        }

        // When inside an envelope, we have the raw content between the opener
        // and end-of-stream (close tag may have been a stop token and never
        // streamed). Parse one or more calls directly.
        //
        // Issue #33: if the incremental path already emitted ToolCallStart
        // for this call (current_tc_name is Some), `emit_envelope_calls`
        // emits ToolCallDelta + ToolCallEnd (with an authoritative name
        // update only if needed) against the staged header rather than a
        // fresh `ToolCall` with a brand-new id.
        if was_envelope.is_some() {
            let parsed_calls = parse_envelope_calls(text.trim(), self.call_counter);
            if !parsed_calls.is_empty() {
                if !canonical_qwen_envelope || !is_canonical_qwen_body(&text) {
                    self.record_parser_recovery();
                }
                let mut out = Vec::new();
                self.emit_envelope_calls(parsed_calls, &mut out);
                self.reset_call_state();
                return out;
            }
        }

        // If a speculative ToolCallStart was emitted before this unparseable
        // flush, emit ToolCallEnd first so `tool_handlers` drops the stale
        // staged entry before any fallback ToolCall or Content is processed.
        let mut out_prefix = Vec::new();
        if self.current_tc_name.is_some() {
            out_prefix.push(DetectorOutput::ToolCallEnd {
                idx: self.call_counter as usize,
            });
        }

        let text = match was_envelope {
            Some(env) => format!("{}{text}", env.open_tag()),
            None => text,
        };

        // Try bare function detection on the remaining buffer.
        // Only if no tool calls were already found (avoid duplicate extraction).
        if !self.has_tool_calls() && !self.emitted_tool_calls {
            let (content, calls) = parse_bare_function_calls(&text);
            if !calls.is_empty() {
                self.record_parser_recovery();
                self.reset_call_state();
                let mut out = out_prefix;
                if let Some(c) = content {
                    out.push(DetectorOutput::Content(c));
                }
                for tc in calls {
                    let idx = self.call_counter as usize;
                    self.call_counter += 1;
                    out.push(DetectorOutput::ToolCall(tc, idx));
                }
                return out;
            }
        }

        // Fallback: JSON tool calls without any XML wrapper.
        // Nemotron-H Super 120B sometimes outputs Hermes-style JSON or JSON
        // in code blocks instead of <tool_call> XML. Catch those here.
        if !self.has_tool_calls() && !self.emitted_tool_calls {
            let json_calls = parse_json_fallback_calls(&text);
            if !json_calls.is_empty() {
                self.record_parser_recovery();
                self.reset_call_state();
                let mut out = out_prefix;
                // Strip matched JSON from content
                let mut clean = text.clone();
                for pattern in extract_json_code_blocks(&text) {
                    clean = clean.replace(&pattern, "");
                }
                let clean = clean.trim().to_string();
                if !clean.is_empty() {
                    out.push(DetectorOutput::Content(clean));
                }
                for tc in json_calls {
                    let idx = self.call_counter as usize;
                    self.call_counter += 1;
                    out.push(DetectorOutput::ToolCall(tc, idx));
                }
                return out;
            }
        }

        if was_envelope.is_some() {
            // A non-empty unclosed Qwen envelope that could not be parsed or
            // recovered above is published as content. Record that parser
            // fallback just as the empty and explicitly closed drop branches
            // do; only a complete canonical inner body may omit the outer
            // close without provenance.
            self.record_parser_recovery();
        }
        self.reset_call_state();
        out_prefix.push(DetectorOutput::Content(text));
        out_prefix
    }

    /// Clear the per-tool-call header bookkeeping.
    pub(super) fn reset_call_state(&mut self) {
        self.current_tc_name = None;
        self.current_tc_id = None;
    }

    pub fn has_tool_calls(&self) -> bool {
        self.call_counter > 0
    }

    /// Returns safe byte length to emit without splitting a partial tag.
    /// Holds back content that could be the start of a tool-call envelope or
    /// bare `<function=` / `<function>` / `<function name=`.
    fn safe_emit_len(&self) -> usize {
        let buf = self.buffer.as_bytes();
        // Check all tag prefixes — don't emit partial matches for any of
        // them, including MiniMax envelope opens (canonical + BPE-split).
        // Without these in the list, a `<minimax:` trailing prefix split
        // across stream chunks gets emitted as content and the detector never
        // sees the complete open tag, leaking the full envelope into `content`.
        // Close tags do not need preserving here: close matching only runs
        // when `active_envelope.is_some()`, where the buffer accumulates the
        // entire inner block until close lands.
        let mut safe = buf.len();
        for tag in [
            b"<tool_call>" as &[u8],
            b"<|tool_call>",
            b"<minimax:tool_call>",
            b"<minimax:_call>",
            b"<function=",
            b"<function>",
            b"<function name=",
            b"call:",
            MISTRAL_TOOL_CALLS_TAG.as_bytes(),
        ] {
            for i in (buf.len().saturating_sub(tag.len() - 1))..buf.len() {
                if tag.starts_with(&buf[i..]) {
                    safe = safe.min(i);
                    break;
                }
            }
        }
        safe
    }
}

/// Locate the earliest bare function opener (`<function=`, `<function>`, or
/// `<function name=`) without matching prose such as Python's
/// `<function foo at 0x...>`.
fn find_bare_function_opener(buf: &str) -> Option<usize> {
    ["<function=", "<function>", "<function name="]
        .into_iter()
        .filter_map(|marker| buf.find(marker))
        .min()
}
