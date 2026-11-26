// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

// Sibling-cluster items hoisted from the original `api.rs`. These uses
// give every sub-file access to helpers that the un-split file took for
// granted via single-module visibility.
use super::super::chat::chat_completions_inner;
use super::super::compact::{
    compact_messages, openai_error_response, openai_error_response_with_param,
};
use super::super::completions::not_supported;
use super::super::inference_impl::{
    extract_thinking, strip_stop_sequences, tokenize_stop_sequences,
};
use super::super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};
use super::super::strip::strip_thinking_tags;
use super::{
    F23ProgressMetrics, F37FailureClass, F39FailureCache, F39PermanentFailureMatch,
    F49DuplicateWrite, bump_f12_tool_call_count, check_loop_watchdog, f23_build_reminder,
    f23_normalize_and_hash, f23_refuse_threshold, f23_score_progress, f23_warn_threshold,
    f31_inject_hard_refusal, f32_reposition_failed_tool_result, f37_classify_failure,
    f39_build_circuit_breaker_banner, f39_build_failure_cache, f39_class_label,
    f39_detect_recent_retries, f39_extract_binary_name, f44_check_permanent_failure,
    f49_build_banner, f49_detect_duplicate_writes, f49_extract_write_path_and_content,
    f50_append_original_error, f60_disable_mtp_for_request, flush_content_sanitizer,
    strip_xml_leaks_from_assistant_content,
};

// Re-export sibling helpers via crate::api::* for short paths.
use super::super::inference_types::*;

pub fn collect_f7_stall_buckets(messages: &[crate::openai::IncomingMessage]) -> F7StallBuckets {
    let mut buckets = F7StallBuckets::new();
    for m in messages {
        if m.role != "assistant" {
            continue;
        }
        let Some(tcs) = &m.tool_calls else {
            continue;
        };
        for tc in tcs {
            let name = tc.function.name.clone();
            let Some(arg) = primary_arg_for_tool(&name, &tc.function.arguments) else {
                continue;
            };
            *buckets.entry((name, arg)).or_insert(0) += 1;
        }
    }
    buckets
}

/// Build the system-reminder text covering all buckets at or above the
/// warn threshold. Returns `None` if no bucket triggers.
pub fn build_f7_stall_reminder(buckets: &F7StallBuckets) -> Option<String> {
    let mut warn_lines: Vec<String> = buckets
        .iter()
        .filter(|(_, c)| **c >= F7_STALL_WARN_THRESHOLD)
        .map(|((name, arg), c)| format!("- {}({}) × {}", name, arg, c))
        .collect();
    if warn_lines.is_empty() {
        return None;
    }
    warn_lines.sort();
    let any_at_refuse = buckets.values().any(|&c| c >= F7_STALL_REFUSE_THRESHOLD);
    let body = warn_lines.join("\n");
    let directive = if any_at_refuse {
        "STOP retrying. Respond to the user with a plain text message explaining what is blocking and what you need from them. Do NOT call any tool again — your next response must contain text only."
    } else {
        "The retries are not making progress. Stop and explain to the user what is blocking and what you need from them. Do not call these tools again with the same arguments."
    };
    Some(format!(
        "\n\n<system-reminder>\nYou have already issued the following tool calls multiple times in this conversation:\n{body}\n{directive}\n</system-reminder>"
    ))
}

/// Append the reminder text to the last user/tool message. If neither
/// exists in the conversation, push a synthetic system message.
pub fn append_f7_reminder_to_last_user(
    messages: &mut Vec<crate::openai::IncomingMessage>,
    reminder: &str,
) {
    for m in messages.iter_mut().rev() {
        if m.role == "user" || m.role == "tool" {
            m.content.text.push_str(reminder);
            return;
        }
    }
    // No user/tool message — push a synthetic system note. Trim
    // leading whitespace so the synthetic message doesn't start with
    // blank lines.
    messages.push(crate::openai::IncomingMessage::synthetic_system(
        reminder.trim_start().to_string(),
    ));
}

/// Prepend a reminder to the system message at position 0. Reminders inside
/// `<tool_response>` envelopes are often treated as ambient observation data
/// (arXiv:2508.15815), whereas prepending to the system message places the
/// signal at high attention. Wrapped in `<flux_runtime_notice>` so
/// position-sensitive templates are not confused by the injected content.
///
/// Idempotent: if a previous notice exists, it is replaced with the fresh
/// content. If no system message exists, a synthetic one is inserted at
/// position 0. Returns `true` only when messages changed.
pub fn prepend_reminder_to_system(
    messages: &mut Vec<crate::openai::IncomingMessage>,
    reminder: &str,
) -> bool {
    // Optional notices may replace existing ranges; leave image-bearing system
    // content intact rather than invalidating its ordered pixel ownership.
    if messages
        .first()
        .is_some_and(|m| m.role == "system" && !m.content.images.is_empty())
    {
        return false;
    }
    let trimmed = reminder.trim_matches(|c: char| c == '\n' || c == ' ');
    let block = format!("<flux_runtime_notice>\n{trimmed}\n</flux_runtime_notice>\n\n");
    let block_core = block.trim_end_matches('\n');
    let has_system_at_zero = messages.first().is_some_and(|m| m.role == "system");
    if has_system_at_zero {
        let txt = &mut messages[0].content.text;
        let existing_span = [
            ("<flux_runtime_notice>", "</flux_runtime_notice>"),
            ("<atlas_runtime_notice>", "</atlas_runtime_notice>"),
        ]
        .into_iter()
        .find_map(|(open, close)| {
            let start = txt.find(open)?;
            let end = txt[start..].find(close)?;
            Some((start, start + end + close.len()))
        });
        if let Some((start, core_end)) = existing_span {
            if &txt[start..core_end] == block_core {
                return false;
            }
            let abs_end = core_end
                + txt[core_end..]
                    .chars()
                    .take_while(|character| *character == '\n')
                    .take(2)
                    .count();
            txt.replace_range(start..abs_end, &block);
            return true;
        }
        let mut new_txt = block;
        new_txt.push_str(txt);
        messages[0].content.text = new_txt;
    } else {
        messages.insert(
            0,
            crate::openai::IncomingMessage::synthetic_system(block.trim_end().to_string()),
        );
    }
    true
}

// ── Environment-facts auto-injection ──
//
// When a binary fails repeatedly with `command not found` / `Exit code 127`,
// that indicates a permanent environment fact (the binary is not installed).
// This pass extracts those facts from message history each turn and prepends
// them to the system message, where models treat instructions as authoritative
// rather than ambient `<tool_response>` data (ChatBug arXiv:2406.12935 /
// arXiv:2508.15815).
//
// Stateless: rebuilt each turn from message history. Threshold of 2 observed
// failures catches immediate retry loops while filtering single transient
// errors.
const F29_MIN_FAILURES_FOR_FACT: u32 = 2;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct F29EnvironmentFact {
    binary: String,
    observed_count: u32,
}

/// Extract a binary name from a `command not found` line. Returns
/// the binary if the line matches the expected error format,
/// otherwise None.
///
/// Patterns matched:
///   "X: command not found"
///   "/bin/bash: line 1: X: command not found"
///   "X: not found"
///   "bash: X: command not found"
///   "/bin/sh: 1: X: not found"
///
/// `Exit code 127` lines without an inline binary name are NOT
/// matched here — `f37_classify_failure` covers those at the
/// per-tool-call level instead.
pub fn f29_extract_binary_from_error_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let suffix = ": command not found";
    let alt_suffix = ": not found";
    let idx = trimmed
        .rfind(suffix)
        .or_else(|| trimmed.rfind(alt_suffix))?;
    let prefix = &trimmed[..idx];
    // Take the LAST whitespace-or-`:`-separated token before the
    // suffix as the binary name. E.g.
    //   "/bin/bash: line 1: cargo" → "cargo"
    //   "cargo" → "cargo"
    let binary = prefix.rsplit([' ', ':', '\t']).next()?.trim();
    // Sanity: binary names are typically [A-Za-z0-9_-]+, no spaces,
    // not too long. Reject if it looks like a path or sentence.
    if binary.is_empty()
        || binary.len() > 40
        || binary.contains('/')
        || !binary
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    {
        return None;
    }
    Some(binary.to_string())
}

pub fn f29_extract_environment_facts(
    messages: &[crate::openai::IncomingMessage],
) -> Vec<F29EnvironmentFact> {
    let mut counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for m in messages {
        if m.role != "tool" {
            continue;
        }
        for line in m.content.text.lines() {
            if let Some(bin) = f29_extract_binary_from_error_line(line) {
                *counts.entry(bin).or_insert(0) += 1;
            }
        }
    }
    let mut facts: Vec<F29EnvironmentFact> = counts
        .into_iter()
        .filter_map(|(binary, observed_count)| {
            if observed_count >= F29_MIN_FAILURES_FOR_FACT {
                Some(F29EnvironmentFact {
                    binary,
                    observed_count,
                })
            } else {
                None
            }
        })
        .collect();
    facts.sort_by(|a, b| a.binary.cmp(&b.binary));
    facts
}

/// Prepend an `<environment_facts>` block to the (existing or new)
/// system message at position 0. Idempotent: if the block already
/// exists in the system message, replace it with the fresh count.
/// Returns true only when messages changed.
pub fn f29_inject_environment_facts(
    messages: &mut Vec<crate::openai::IncomingMessage>,
    facts: &[F29EnvironmentFact],
) -> bool {
    if facts.is_empty() {
        return false;
    }
    let lines: Vec<String> = facts
        .iter()
        .map(|f| {
            format!(
                "- `{}` (returned 'command not found' {} time{})",
                f.binary,
                f.observed_count,
                if f.observed_count == 1 { "" } else { "s" }
            )
        })
        .collect();
    let block = format!(
        "<environment_facts>\nThe following commands are NOT available in this environment. Do NOT attempt to use them — the failure is permanent and retrying with cosmetic variations (mkdir prefix, cd path, flag order) will not change the outcome:\n{}\n</environment_facts>\n\n",
        lines.join("\n")
    );
    let block_core = block.trim_end_matches('\n');

    // Optional idempotent rewrites must not move/delete embedded image positions.
    if messages
        .first()
        .is_some_and(|m| m.role == "system" && !m.content.images.is_empty())
    {
        return false;
    }
    // Locate or synthesise system message at position 0.
    let has_system_at_zero = messages.first().is_some_and(|m| m.role == "system");
    if has_system_at_zero {
        // Idempotent replace: if a previous block exists, swap it.
        let txt = &mut messages[0].content.text;
        if let Some(start) = txt.find("<environment_facts>")
            && let Some(end) = txt[start..].find("</environment_facts>")
        {
            let core_end = start + end + "</environment_facts>".len();
            if &txt[start..core_end] == block_core {
                return false;
            }
            let abs_end = core_end
                + txt[core_end..]
                    .chars()
                    .take_while(|character| *character == '\n')
                    .take(2)
                    .count();
            txt.replace_range(start..abs_end, &block);
            return true;
        }
        // No prior block — prepend.
        let mut new_txt = block;
        new_txt.push_str(txt);
        messages[0].content.text = new_txt;
    } else {
        messages.insert(
            0,
            crate::openai::IncomingMessage::synthetic_system(block.trim_end().to_string()),
        );
    }
    true
}

// Test whether the most recent `role: tool` message in the conversation is an
// error result. Used to gate auto-disable-thinking and failure-recovery
// behaviour. Checks for `[tool error]`, command-not-found patterns, and raw
// client runtime errors (TypeError, ERR_INVALID_ARG_VALUE, "cannot access",
// "No such file or directory", "already exists", "must read file ... before
// overwriting", and bare "Error:" prefix) forwarded over the OpenAI API.
pub fn recent_message_is_tool_error(messages: &[crate::openai::IncomingMessage]) -> bool {
    for m in messages.iter().rev() {
        if m.role == "tool" {
            let t = &m.content.text;
            return f28_text_looks_like_error(t);
        }
    }
    false
}

pub fn f28_text_looks_like_error(t: &str) -> bool {
    if t.starts_with("[tool error]") || t.starts_with("Error:") {
        return true;
    }
    // F37/F41 expanded permanent-failure patterns. Order: most
    // common first for short-circuit.
    const SIGNATURES: &[&str] = &[
        "Exit code 127",
        "command not found",
        ": not found",
        "Permission denied",
        "[flux-stall-guard]",
        "[flux-permanent-failure]",
        "<flux_circuit_breaker>",
        "[atlas-stall-guard]",
        "[atlas-permanent-failure]",
        "<atlas_circuit_breaker>",
        // F41 (opencode coverage)
        "TypeError",
        "ERR_INVALID_ARG_VALUE",
        "cannot access",
        "No such file or directory",
        // F37 (env-state failures observed in opencode)
        "already exists",
        "cannot be run on existing",
        "before overwriting it",
        "ENOENT",
        "EISDIR",
        "EACCES",
    ];
    SIGNATURES.iter().any(|s| t.contains(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── f29_extract_binary_from_error_line ────────────────────────

    #[test]
    fn binary_from_command_not_found() {
        assert_eq!(
            f29_extract_binary_from_error_line("/bin/sh: 1: cargo: command not found"),
            Some("cargo".to_string())
        );
    }

    #[test]
    fn binary_from_bare_not_found() {
        assert_eq!(
            f29_extract_binary_from_error_line("npm: not found"),
            Some("npm".to_string())
        );
    }

    #[test]
    fn binary_from_bash_format() {
        assert_eq!(
            f29_extract_binary_from_error_line("/bin/bash: line 1: cargo: command not found"),
            Some("cargo".to_string())
        );
    }

    #[test]
    fn binary_no_suffix_returns_none() {
        assert_eq!(
            f29_extract_binary_from_error_line("just some random text"),
            None
        );
    }

    #[test]
    fn binary_too_long_rejected() {
        // 50-char "binary name" — exceeds 40-char sanity cap.
        let line = format!("{}: command not found", "x".repeat(50));
        assert_eq!(f29_extract_binary_from_error_line(&line), None);
    }

    #[test]
    fn binary_with_path_rejected() {
        // Looks like a path, not a bare binary name.
        assert_eq!(
            f29_extract_binary_from_error_line("/usr/bin/foo: command not found"),
            None
        );
    }

    #[test]
    fn binary_accepts_dotted_name() {
        assert_eq!(
            f29_extract_binary_from_error_line("python3.11: command not found"),
            Some("python3.11".to_string())
        );
    }

    // ── f28_text_looks_like_error ─────────────────────────────────

    #[test]
    fn f28_tool_error_prefix() {
        assert!(f28_text_looks_like_error(
            "[tool error] something went wrong"
        ));
        assert!(f28_text_looks_like_error("Error: bad input"));
    }

    #[test]
    fn f28_exit_code_signature() {
        assert!(f28_text_looks_like_error("Build failed. Exit code 127"));
    }

    #[test]
    fn f28_permission_signature() {
        assert!(f28_text_looks_like_error("Permission denied: /tmp/x"));
        assert!(f28_text_looks_like_error("open EACCES"));
    }

    #[test]
    fn f28_opencode_signature() {
        // F41 added opencode-specific patterns.
        assert!(f28_text_looks_like_error("TypeError: cannot read property"));
        assert!(f28_text_looks_like_error("ERR_INVALID_ARG_VALUE"));
    }

    #[test]
    fn f28_clean_output_is_not_error() {
        assert!(!f28_text_looks_like_error("OK"));
        assert!(!f28_text_looks_like_error("Build successful"));
        assert!(!f28_text_looks_like_error(""));
    }
}
