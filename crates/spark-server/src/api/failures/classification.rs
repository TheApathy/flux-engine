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
    F29EnvironmentFact, F39FailureCache, F39PermanentFailureMatch, F49DuplicateWrite,
    append_f7_reminder_to_last_user, build_f7_stall_reminder, bump_f12_tool_call_count,
    check_loop_watchdog, collect_f7_stall_buckets, f28_text_looks_like_error,
    f29_extract_binary_from_error_line, f29_extract_environment_facts,
    f29_inject_environment_facts, f31_inject_hard_refusal, f32_reposition_failed_tool_result,
    f39_build_circuit_breaker_banner, f39_build_failure_cache, f39_class_label,
    f39_detect_recent_retries, f39_extract_binary_name, f44_check_permanent_failure,
    f49_build_banner, f49_detect_duplicate_writes, f49_extract_write_path_and_content,
    f50_append_original_error, f60_disable_mtp_for_request, flush_content_sanitizer,
    prepend_reminder_to_system, recent_message_is_tool_error,
    strip_xml_leaks_from_assistant_content,
};

// Re-export sibling helpers via crate::api::* for short paths.
use super::super::inference_types::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F37FailureClass {
    BinaryMissing,
    AlreadyExists,
    PermissionDenied,
    NotFound,
    InvalidArgument,
    StallGuard,
}

pub fn f37_classify_failure(t: &str) -> Option<F37FailureClass> {
    if t.contains("[flux-stall-guard]")
        || t.contains("[flux-permanent-failure]")
        || t.contains("<flux_circuit_breaker>")
        || t.contains("[atlas-stall-guard]")
        || t.contains("[atlas-permanent-failure]")
        || t.contains("<atlas_circuit_breaker>")
    {
        return Some(F37FailureClass::StallGuard);
    }
    if t.contains("command not found") || t.contains(": not found") || t.contains("Exit code 127") {
        return Some(F37FailureClass::BinaryMissing);
    }
    if t.contains("already exists")
        || t.contains("cannot be run on existing")
        || t.contains("destination") && t.contains("exists")
    {
        return Some(F37FailureClass::AlreadyExists);
    }
    if t.contains("Permission denied") || t.contains("EACCES") {
        return Some(F37FailureClass::PermissionDenied);
    }
    if t.contains("No such file or directory")
        || t.contains("ENOENT")
        || t.contains("cannot access")
    {
        return Some(F37FailureClass::NotFound);
    }
    if t.contains("TypeError")
        || t.contains("ERR_INVALID_ARG_VALUE")
        || t.contains("before overwriting it")
    {
        return Some(F37FailureClass::InvalidArgument);
    }
    None
}

// ── Per-conversation tool-progress tracker ──
//
// Alongside per-tool stall bucketing and per-response hash deduplication,
// this axis scores whether tool calls across the conversation are making
// forward progress:
//   - is_error / "command not found" / "Exit code"  → -1
//   - content hash matches a prior tool result      → -1
//   - all empty / no-op (e.g. mkdir on existing dir)→  0
//   - otherwise (novel result)                       → +1
//
// Default thresholds: warn at `attempts >= 4 && score <= 0`, refuse at
// `attempts >= 6 && score <= 0`. Configurable via `FLUX_F23_WARN_ATTEMPTS`
// and `FLUX_F23_REFUSE_ATTEMPTS` (with `ATLAS_*` fallbacks).
const F23_WARN_ATTEMPTS_DEFAULT: u32 = 4;
const F23_REFUSE_ATTEMPTS_DEFAULT: u32 = 6;

#[derive(Debug, Clone, Copy)]
pub struct F23ProgressMetrics {
    pub score: i32,
    pub attempts: u32,
}

pub fn f23_normalize_and_hash(content: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut h = DefaultHasher::new();
    // Lightweight normalisation: strip trailing whitespace from each line.
    let cleaned: String = content
        .lines()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n");
    h.write(cleaned.as_bytes());
    h.finish()
}

pub fn f23_score_progress(messages: &[crate::openai::IncomingMessage]) -> F23ProgressMetrics {
    let mut score: i32 = 0;
    let mut attempts: u32 = 0;
    let mut seen_hashes: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    // Also track (tool, primary_arg) collisions on the assistant side so
    // repeated identical tool invocations that return distinct success text
    // are still penalized as stalled progress.
    let mut seen_call_keys: std::collections::HashMap<(String, String), u32> =
        std::collections::HashMap::new();
    // Stronger penalty (-2 from the 3rd occurrence onward) when the same
    // (tool, primary_arg) call repeats with an identical trivial success result.
    let mut last_result_for_call: std::collections::HashMap<(String, String), u64> =
        std::collections::HashMap::new();
    let mut last_call_key_seen: Option<(String, String)> = None;
    for m in messages {
        if m.role == "assistant"
            && let Some(tcs) = &m.tool_calls
        {
            attempts = attempts.saturating_add(tcs.len() as u32);
            // F40: count call-key collisions.
            for tc in tcs {
                let name = tc.function.name.clone();
                let Some(arg) = primary_arg_for_tool(&name, &tc.function.arguments) else {
                    continue;
                };
                let key = (name.clone(), arg.clone());
                let prior = *seen_call_keys
                    .entry(key.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                if prior > 1 {
                    score -= 1;
                }
                // F53: remember the most-recent assistant tool_call
                // key so we can pair it with its result below.
                last_call_key_seen = Some(key);
            }
        }
        if m.role == "tool" {
            let content = &m.content.text;
            // 1) Explicit error markers (F6 prepends "[tool error]\n").
            if content.starts_with("[tool error]")
                || content.contains("Exit code 127")
                || content.contains("command not found")
                || content.contains("Permission denied")
                || content.contains("ENOENT")
                || content.contains("EISDIR")
            {
                score -= 1;
                last_call_key_seen = None;
                continue;
            }
            // 2) Repeated content (normalised hash).
            let h = f23_normalize_and_hash(content);
            let prior = seen_hashes.get(&h).copied().unwrap_or(0);
            seen_hashes.insert(h, prior + 1);
            // F53 (2026-04-27): for trivial successful results
            // (empty / Done / OK), pair the result with the most
            // recent assistant call key. If the same (call, result)
            // pair has been seen 3+ times, penalise -2 (above and
            // beyond F40's call-key collision penalty of -1). This
            // catches repeated "mkdir succeeds with no output"
            // patterns that other detectors silently neutral on.
            let stripped = content.trim();
            let trivial_success = stripped.is_empty() || stripped == "Done" || stripped == "OK";
            if trivial_success {
                if let Some(call_key) = last_call_key_seen.take() {
                    let prev = last_result_for_call.get(&call_key).copied();
                    last_result_for_call.insert(call_key.clone(), h);
                    if let Some(p) = prev
                        && p == h
                    {
                        // same call, same trivial result, recurring →
                        // -2 from the 3rd occurrence (the 1st recur is the 2nd
                        // call; the 2nd recur is the 3rd — start penalising at
                        // the 2nd recur to align with "3+ occurrences").
                        let count = seen_call_keys.get(&call_key).copied().unwrap_or(1);
                        if count >= 3 {
                            score -= 2;
                        }
                    }
                }
                continue; // 0 from baseline; F53 already adjusted score
            }
            if prior > 0 {
                score -= 1;
                last_call_key_seen = None;
                continue;
            }
            // 3) Novel useful result.
            score += 1;
            last_call_key_seen = None;
        }
    }
    F23ProgressMetrics { score, attempts }
}

pub fn f23_warn_threshold() -> u32 {
    std::env::var("FLUX_F23_WARN_ATTEMPTS")
        .ok()
        .or_else(|| std::env::var("ATLAS_F23_WARN_ATTEMPTS").ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(F23_WARN_ATTEMPTS_DEFAULT)
}

pub fn f23_refuse_threshold() -> u32 {
    std::env::var("FLUX_F23_REFUSE_ATTEMPTS")
        .ok()
        .or_else(|| std::env::var("ATLAS_F23_REFUSE_ATTEMPTS").ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(F23_REFUSE_ATTEMPTS_DEFAULT)
}

pub fn f23_build_reminder(metrics: F23ProgressMetrics) -> Option<String> {
    let warn = f23_warn_threshold();
    let refuse = f23_refuse_threshold();
    if metrics.attempts >= refuse && metrics.score <= 0 {
        Some(format!(
            "\n\n<system-reminder>\n\
             [STALL GUARD] You have called {} tools without making progress \
             (progress_score={}). STOP. Do NOT call any more tools. Reply \
             to the user with a plain-text explanation of what is blocking \
             and what you need from them. Subsequent tool calls will be \
             refused.\n\
             </system-reminder>",
            metrics.attempts, metrics.score
        ))
    } else if metrics.attempts >= warn && metrics.score <= 0 {
        Some(format!(
            "\n\n<system-reminder>\n\
             [STALL GUARD] You have called {} tools without making progress \
             (progress_score={}). The retries are not advancing the task. \
             Stop and explain to the user what is blocking. Do not retry \
             the same approach.\n\
             </system-reminder>",
            metrics.attempts, metrics.score
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_stall_guard() {
        assert_eq!(
            f37_classify_failure("[flux-stall-guard] something"),
            Some(F37FailureClass::StallGuard)
        );
        assert_eq!(
            f37_classify_failure("[flux-permanent-failure] xx"),
            Some(F37FailureClass::StallGuard)
        );
        assert_eq!(
            f37_classify_failure("<flux_circuit_breaker> repeat"),
            Some(F37FailureClass::StallGuard)
        );
    }

    #[test]
    fn classify_binary_missing() {
        assert_eq!(
            f37_classify_failure("bash: foo: command not found"),
            Some(F37FailureClass::BinaryMissing)
        );
        assert_eq!(
            f37_classify_failure("zsh: bar: not found"),
            Some(F37FailureClass::BinaryMissing)
        );
        assert_eq!(
            f37_classify_failure("Exit code 127"),
            Some(F37FailureClass::BinaryMissing)
        );
    }

    #[test]
    fn classify_permission_denied() {
        assert_eq!(
            f37_classify_failure("Permission denied"),
            Some(F37FailureClass::PermissionDenied)
        );
        assert_eq!(
            f37_classify_failure("open: EACCES"),
            Some(F37FailureClass::PermissionDenied)
        );
    }

    #[test]
    fn classify_not_found() {
        assert_eq!(
            f37_classify_failure("No such file or directory"),
            Some(F37FailureClass::NotFound)
        );
        assert_eq!(
            f37_classify_failure("readdir: ENOENT"),
            Some(F37FailureClass::NotFound)
        );
        assert_eq!(
            f37_classify_failure("cannot access /tmp/x"),
            Some(F37FailureClass::NotFound)
        );
    }

    #[test]
    fn classify_already_exists() {
        assert_eq!(
            f37_classify_failure("file already exists"),
            Some(F37FailureClass::AlreadyExists)
        );
        assert_eq!(
            f37_classify_failure("cannot be run on existing directory"),
            Some(F37FailureClass::AlreadyExists)
        );
    }

    #[test]
    fn classify_invalid_argument() {
        assert_eq!(
            f37_classify_failure("TypeError: bad value"),
            Some(F37FailureClass::InvalidArgument)
        );
        assert_eq!(
            f37_classify_failure("Error [ERR_INVALID_ARG_VALUE]"),
            Some(F37FailureClass::InvalidArgument)
        );
    }

    #[test]
    fn classify_unknown_returns_none() {
        assert_eq!(f37_classify_failure(""), None);
        assert_eq!(f37_classify_failure("OK"), None);
        assert_eq!(f37_classify_failure("Random output"), None);
    }

    #[test]
    fn normalize_hash_stable() {
        // Same string → same hash, twice.
        let h1 = f23_normalize_and_hash("hello world");
        let h2 = f23_normalize_and_hash("hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn normalize_hash_differs_for_different_content() {
        let a = f23_normalize_and_hash("hello world");
        let b = f23_normalize_and_hash("goodbye world");
        assert_ne!(a, b);
    }

    #[test]
    fn build_reminder_returns_none_when_under_warn() {
        // attempts=0, score=0: no reminder.
        let m = F23ProgressMetrics {
            score: 0,
            attempts: 0,
        };
        assert!(f23_build_reminder(m).is_none());
    }

    #[test]
    fn build_reminder_returns_none_when_score_positive() {
        // Score > 0 means progress — no reminder regardless of attempts.
        let m = F23ProgressMetrics {
            score: 5,
            attempts: 100,
        };
        assert!(f23_build_reminder(m).is_none());
    }

    #[test]
    fn build_reminder_warns_at_warn_threshold() {
        // Default warn=4: 4 attempts, no progress → Some(warn).
        let m = F23ProgressMetrics {
            score: 0,
            attempts: 4,
        };
        let out = f23_build_reminder(m);
        assert!(out.is_some());
        assert!(out.unwrap().contains("STALL GUARD"));
    }

    #[test]
    fn build_reminder_refuses_at_refuse_threshold() {
        // Default refuse=6: 6 attempts, no progress → Some(refuse).
        let m = F23ProgressMetrics {
            score: 0,
            attempts: 6,
        };
        let out = f23_build_reminder(m);
        assert!(out.is_some());
        let body = out.unwrap();
        assert!(body.contains("STALL GUARD"));
        assert!(body.contains("STOP"), "refuse-level message includes STOP");
    }
}
