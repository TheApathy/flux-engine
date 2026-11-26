// SPDX-License-Identifier: AGPL-3.0-only

//! Complete native-envelope publication, shared by both streaming modes.

use super::streaming::EnvelopeKind;
use super::{DetectorOutput, StreamingToolDetector, glm_xml, glm_xml_scan};

pub(super) enum NativeStep {
    NotNative,
    Consumed,
    Pending,
}

impl StreamingToolDetector {
    pub(super) fn process_glm_native(&mut self, outputs: &mut Vec<DetectorOutput>) -> NativeStep {
        if self.active_envelope.is_none() {
            let Some(start) = glm_xml_scan::first_native_opener(&self.buffer) else {
                return NativeStep::NotNative;
            };
            let before = self.buffer[..start].to_string();
            self.buffer = self.buffer[start + glm_xml_scan::OPEN.len()..].to_string();
            self.active_envelope = Some(EnvelopeKind::QwenOrHermes);
            if !before.is_empty() {
                outputs.push(DetectorOutput::Content(before));
            }
            return NativeStep::Consumed;
        }
        if self.active_envelope != Some(EnvelopeKind::QwenOrHermes) {
            return NativeStep::NotNative;
        }
        if !glm_xml_scan::native_prefix(&self.buffer) {
            return NativeStep::NotNative;
        }
        let Some(end) = glm_xml_scan::native_close(&self.buffer) else {
            return NativeStep::Pending;
        };
        // Any completed GLM-native envelope is a foreign-format path when the
        // configured contract is canonical Qwen, whether it parses or is
        // republished as content.
        self.record_parser_recovery();
        let parsed = glm_xml::parse_body(&self.buffer[..end]);
        if let Some(call) = parsed {
            outputs.push(DetectorOutput::ToolCall(call, self.call_counter as usize));
            self.call_counter += 1;
            self.emitted_tool_calls = true;
        } else {
            let body = &self.buffer[..end];
            outputs.push(DetectorOutput::Content(format!(
                "<tool_call>{body}</tool_call>"
            )));
        }
        self.buffer = self.buffer[end + glm_xml_scan::CLOSE.len()..].to_string();
        self.active_envelope = None;
        self.reset_call_state();
        NativeStep::Consumed
    }
}
