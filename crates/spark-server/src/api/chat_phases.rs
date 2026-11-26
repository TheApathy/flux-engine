// SPDX-License-Identifier: AGPL-3.0-only

//! Helper functions extracted from `chat::chat_completions_inner`. These
//! are the F1..F60 cross-turn agentic guards — each one inspects and
//! mutates `req.messages` (and occasionally returns a flag the caller
//! uses for log-line correlation). Keeping them in a sibling file makes
//! `chat.rs` fit the 500-LoC cap and gives each guard a clear top-level
//! seam for future testing.

use axum::http::StatusCode;
use axum::response::Response;

use crate::openai::ChatCompletionRequest;

use super::compact::{openai_error_response, openai_error_response_with_param};
use super::failures::{
    F23ProgressMetrics, append_f7_reminder_to_last_user, build_f7_stall_reminder,
    collect_f7_stall_buckets, f23_build_reminder, f23_score_progress,
    f29_extract_environment_facts, f29_inject_environment_facts, f31_inject_hard_refusal,
    f32_reposition_failed_tool_result, f39_build_circuit_breaker_banner, f39_detect_recent_retries,
    f49_build_banner, f49_detect_duplicate_writes, f50_append_original_error,
    prepend_reminder_to_system, recent_message_is_tool_error,
};
use super::sanitizer::{F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD};

/// Opt-in gate (`FLUX_AGENT_GUARDS=1` / `ATLAS_AGENT_GUARDS=1`, default `false`)
/// for legacy cross-turn prompt/thinking rewrites (`F7..F52`, `F28`, `hint_injector`,
/// `cwd_hint`, `task_pin`). Keeping these off by default preserves exact OpenAI
/// wire fidelity, KV prefix-cache continuity, and post-error `<think>` reasoning
/// on par with stock `vLLM` / `SGLang`.
pub(crate) fn agent_guards_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        ["FLUX_AGENT_GUARDS", "ATLAS_AGENT_GUARDS"]
            .into_iter()
            .find_map(|k| std::env::var(k).ok())
            .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true"))
    })
}

/// Validate the OpenAI input contract: messages length, max_tokens > 0,
/// temperature/top_p ranges, tool_choice mode/required compatibility.
/// Returns `Err(Response)` for fail-fast 400 paths so the caller can
/// `?` directly into a Response.
#[allow(clippy::result_large_err)]
pub(super) fn validate_input(req: &ChatCompletionRequest) -> Result<(), Response> {
    if let Some(crate::openai::ResponseFormat::JsonSchema { ref json_schema }) = req.response_format
        && json_schema.strict
        && let Err(err) = crate::api::schema_check::validate_strict_json_schema(&json_schema.schema)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            err.to_string(),
            Some("response_format.json_schema.schema"),
            Some("invalid_json_schema"),
        ));
    }
    if req.messages.is_empty() {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "messages must contain at least one message".into(),
            Some("messages"),
            None,
        ));
    }
    if req.messages.len() > 16384 {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "messages array exceeds maximum length (16384)".into(),
            Some("messages"),
            None,
        ));
    }
    if req.stop.len() > 64 {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "stop must contain at most 64 sequences".into(),
            Some("stop"),
            None,
        ));
    }
    if req.stop.iter().any(String::is_empty) {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "stop sequences must not be empty".into(),
            Some("stop"),
            None,
        ));
    }
    if let Some(t) = req.temperature
        && (!(0.0..=2.0).contains(&t))
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "temperature must be between 0 and 2".into(),
            Some("temperature"),
            None,
        ));
    }
    if let Some(p) = req.top_p
        && !(p > 0.0 && p <= 1.0)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "top_p must be between 0 (exclusive) and 1".into(),
            Some("top_p"),
            None,
        ));
    }
    if let Some(top_n_sigma) = req.top_n_sigma
        && !(top_n_sigma.is_finite() && top_n_sigma >= 0.0)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "top_n_sigma must be finite and non-negative".into(),
            Some("top_n_sigma"),
            None,
        ));
    }
    if let Some(min_p) = req.min_p
        && !(0.0..=1.0).contains(&min_p)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "min_p must be between 0 and 1".into(),
            Some("min_p"),
            None,
        ));
    }
    if let Some(repetition_penalty) = req.repetition_penalty
        && !(repetition_penalty.is_finite() && repetition_penalty > 0.0)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "repetition_penalty must be finite and greater than 0".into(),
            Some("repetition_penalty"),
            None,
        ));
    }
    for (name, penalty) in [
        ("presence_penalty", req.presence_penalty),
        ("frequency_penalty", req.frequency_penalty),
    ] {
        if let Some(value) = penalty
            && !(-2.0..=2.0).contains(&value)
        {
            return Err(openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                format!("{name} must be between -2.0 and 2.0"),
                Some(name),
                None,
            ));
        }
    }
    if req.n == 0 || req.n > 128 {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "n must be between 1 and 128".into(),
            Some("n"),
            None,
        ));
    }
    if req.max_tokens == 0 {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "max_tokens must be at least 1".into(),
            Some("max_tokens"),
            None,
        ));
    }
    if let Some(logit_bias) = &req.logit_bias {
        for (token_id, bias) in logit_bias {
            if token_id.parse::<u32>().is_err() {
                return Err(openai_error_response_with_param(
                    StatusCode::BAD_REQUEST,
                    format!("logit_bias key '{token_id}' must be a token ID"),
                    Some("logit_bias"),
                    None,
                ));
            }
            if !(-100.0..=100.0).contains(bias) {
                return Err(openai_error_response_with_param(
                    StatusCode::BAD_REQUEST,
                    "logit_bias values must be between -100 and 100".into(),
                    Some("logit_bias"),
                    None,
                ));
            }
        }
    }
    if let Some(crate::tool_parser::ToolChoice::Mode(ref s)) = req.tool_choice {
        if !["auto", "none", "required"].contains(&s.as_str()) {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "Invalid tool_choice value: '{s}'. Must be 'auto', 'none', 'required', or a function object."
                ),
            ));
        }
        if s == "required" && req.tools.as_ref().is_none_or(|t| t.is_empty()) {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                "tool_choice is 'required' but no tools were provided".into(),
            ));
        }
    }
    if let Some(crate::tool_parser::ToolChoice::Specific { ref function }) = req.tool_choice {
        let Some(tools) = req.tools.as_ref().filter(|tools| !tools.is_empty()) else {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                "specific tool_choice was provided but no tools were provided".into(),
            ));
        };
        if !tools.iter().any(|tool| tool.function.name == function.name) {
            return Err(openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                format!("tool_choice selected unknown function '{}'.", function.name),
                Some("tool_choice"),
                Some("tool_not_found"),
            ));
        }
    }
    Ok(())
}

/// Apply the cross-turn F-feature guards (F7, F23, F30, F31, F32, F39,
/// F49, F50, F35, F52, F29) to `req.messages`. Each guard inspects the
/// conversation history and mutates messages by injecting reminders /
/// circuit-breaker banners / synthesised tool_results. Returns the
/// computed `F23ProgressMetrics` so the caller can pass it through to
/// downstream phases that also key on it.
pub(super) fn apply_failure_guards(req: &mut ChatCompletionRequest) -> F23ProgressMetrics {
    let mut system_notices: Vec<String> = Vec::new();
    let mut f23_triggered = false;
    let mut f39_triggered = false;
    let mut f49_triggered = false;
    let mut f35_triggered = false;

    // Cross-turn tool-arg-path stall guard.
    let stall_buckets = collect_f7_stall_buckets(&req.messages);
    if let Some(reminder) = build_f7_stall_reminder(&stall_buckets) {
        tracing::warn!(
            buckets_at_warn = stall_buckets
                .iter()
                .filter(|(_, c)| **c >= F7_STALL_WARN_THRESHOLD)
                .count(),
            buckets_at_refuse = stall_buckets
                .iter()
                .filter(|(_, c)| **c >= F7_STALL_REFUSE_THRESHOLD)
                .count(),
            "tool-arg stall guard: injecting reminder into last user/tool message"
        );
        append_f7_reminder_to_last_user(&mut req.messages, &reminder);
        req.record_intervention("F7", "append_stall_reminder");
    }

    // Per-conversation progress tracker.
    let f23_metrics = f23_score_progress(&req.messages);
    if let Some(reminder) = f23_build_reminder(f23_metrics) {
        tracing::warn!(
            attempts = f23_metrics.attempts,
            score = f23_metrics.score,
            "progress tracker: queueing stall reminder for system message"
        );
        system_notices.push(reminder);
        f23_triggered = true;
    }

    // Synthesise [flux-stall-guard] tool_result at refuse threshold.
    if f31_inject_hard_refusal(&mut req.messages, f23_metrics) {
        tracing::warn!(
            attempts = f23_metrics.attempts,
            score = f23_metrics.score,
            "stall guard: injected synthesised [flux-stall-guard] tool_result"
        );
        req.record_intervention("F31", "inject_synthetic_tool_error");
    }

    // Duplicate failed tool_result at conversation tail.
    if f32_reposition_failed_tool_result(&mut req.messages) {
        tracing::info!(
            "failure recovery: duplicated most-recent failed tool_result at conversation tail"
        );
        req.record_intervention("F32", "duplicate_failed_tool_result");
    }

    // Cross-turn permanent-failure circuit breaker.
    let f39_matches = f39_detect_recent_retries(&req.messages);
    if !f39_matches.is_empty() {
        tracing::warn!(
            n_matches = f39_matches.len(),
            sample = ?f39_matches.iter().take(3).map(|m| (&m.tool, &m.primary_arg, m.class, m.prior_failure_count)).collect::<Vec<_>>(),
            "permanent-failure circuit breaker: model is retrying calls with permanent prior failures"
        );
        system_notices.push(f39_build_circuit_breaker_banner(&f39_matches));
        f39_triggered = true;
    }

    // Duplicate-write detector and original-error re-surfacing.
    let f49_hits = f49_detect_duplicate_writes(&req.messages);
    if !f49_hits.is_empty() {
        tracing::warn!(
            n_hits = f49_hits.len(),
            sample = ?f49_hits.iter().take(3).map(|h| (&h.file_path, h.prior_count)).collect::<Vec<_>>(),
            "duplicate-write detector: model is rewriting same file with identical content"
        );
        system_notices.push(f49_build_banner(&f49_hits));
        f49_triggered = true;
        if f50_append_original_error(&mut req.messages) {
            tracing::info!(
                "duplicate-write recovery: appended original [tool error] at conversation tail"
            );
            req.record_intervention("F50", "append_original_tool_error");
        }
    }

    // Turn-conditional failure-recovery clause with concrete fallbacks.
    if recent_message_is_tool_error(&req.messages) {
        let f35_clause = "<failure_recovery>\nThe previous tool call failed with an error. \
             That failure is a deterministic fact about this environment — \
             retrying with cosmetic variations (different mkdir prefix, \
             different cd path, different flag order) cannot change the \
             outcome.\n\n\
             Choose ONE of these concrete next steps:\n\
             (a) If the failed tool was Bash and the binary is missing, use the Write tool to create the files MANUALLY (write the Cargo.toml, source files, etc. directly via Write — do NOT call the missing binary again).\n\
             (b) If the failure is a path/permissions issue, try Bash with a substantively different command (different directory, different flags, different invocation).\n\
             (c) If neither (a) nor (b) is feasible, reply to the user in PLAIN TEXT (no tool call) explaining specifically what is blocking and what dependency or permission they need to provide.\n\
             Do NOT emit a generic \"please clarify your request\" question — the user already gave the request. Pick (a), (b), or (c) and execute it.\n\
             </failure_recovery>";
        system_notices.push(f35_clause.to_string());
        f35_triggered = true;
    }

    // Combine all active system notices into a single `<flux_runtime_notice>`
    // block so multiple guards firing on the same turn never overwrite each other.
    if !system_notices.is_empty() {
        let combined = system_notices.join("\n\n");
        if prepend_reminder_to_system(&mut req.messages, &combined) {
            if f23_triggered {
                req.record_intervention("F23", "trigger_progress_guard");
                req.record_intervention("F30", "prepend_progress_reminder");
            }
            if f39_triggered {
                req.record_intervention("F39", "prepend_permanent_failure_circuit_breaker");
            }
            if f49_triggered {
                req.record_intervention("F49", "prepend_duplicate_write_banner");
            }
            if f35_triggered {
                tracing::debug!(
                    "failure recovery: prepended failure_recovery clause (most recent message is tool error)"
                );
                req.record_intervention("F35", "trigger_failure_recovery");
                req.record_intervention("F52", "prepend_failure_recovery_clause");
            }
        }
    }

    // Scan tool-result history for repeated `command not found` failures
    // and inject an `<environment_facts>` block into the system message.
    let env_facts = f29_extract_environment_facts(&req.messages);
    if !env_facts.is_empty() {
        tracing::info!(
            facts_count = env_facts.len(),
            "environment facts: injecting environment_facts block into system message"
        );
        if f29_inject_environment_facts(&mut req.messages, &env_facts) {
            req.record_intervention("F29", "inject_environment_facts");
        }
    }

    f23_metrics
}

#[cfg(test)]
mod tool_choice_validation_tests {
    use super::validate_input;
    use axum::http::StatusCode;

    fn request(value: serde_json::Value) -> crate::openai::ChatCompletionRequest {
        serde_json::from_value(value).expect("valid request fixture")
    }

    #[test]
    fn specific_choice_requires_a_matching_declared_tool() {
        let absent = request(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "weather"}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }));
        assert_eq!(
            validate_input(&absent).unwrap_err().status(),
            StatusCode::BAD_REQUEST
        );

        let unknown = request(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }));
        assert_eq!(
            validate_input(&unknown).unwrap_err().status(),
            StatusCode::BAD_REQUEST
        );

        let matching = request(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }));
        assert!(validate_input(&matching).is_ok());
    }

    #[test]
    fn numeric_sampling_contract_rejects_nonfinite_and_out_of_range_values() {
        let mut value = request(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        }));

        for invalid in [-1.0, 2.1, f32::NAN, f32::INFINITY] {
            value.temperature = Some(invalid);
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        value.temperature = None;
        for invalid in [0.0, -0.1, 1.1, f32::NAN, f32::INFINITY] {
            value.top_p = Some(invalid);
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }
    }

    #[test]
    fn logit_bias_rejects_invalid_keys_and_values() {
        let mut value = request(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        }));
        value.logit_bias = Some(std::collections::HashMap::from([
            ("0".to_string(), -100.0),
            (u32::MAX.to_string(), 100.0),
        ]));
        assert!(validate_input(&value).is_ok());

        for key in ["not-token", "-1", "4294967296"] {
            value.logit_bias = Some(std::collections::HashMap::from([(key.to_string(), 0.0)]));
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        for bias in [-100.1, 100.1, f32::NAN, f32::INFINITY] {
            value.logit_bias = Some(std::collections::HashMap::from([("0".to_string(), bias)]));
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }
    }

    #[test]
    fn extended_sampling_contract_is_fail_closed() {
        let mut value = request(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        }));
        for invalid in [-0.1, f32::NAN, f32::INFINITY] {
            value.top_n_sigma = Some(invalid);
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        value.top_n_sigma = None;
        for invalid in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
            value.min_p = Some(invalid);
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        value.min_p = None;
        for invalid in [0.0, -0.1, f32::NAN, f32::INFINITY] {
            value.repetition_penalty = Some(invalid);
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        value.repetition_penalty = Some(f32::MIN_POSITIVE);
        assert!(validate_input(&value).is_ok());

        for invalid in [-2.1, 2.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            value.presence_penalty = Some(invalid);
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
            value.presence_penalty = None;
            value.frequency_penalty = Some(invalid);
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
            value.frequency_penalty = None;
        }
        value.presence_penalty = Some(-2.0);
        value.frequency_penalty = Some(2.0);
        assert!(validate_input(&value).is_ok());

        for invalid_n in [0, 129] {
            value.n = invalid_n;
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }
        value.n = 1;
        assert!(validate_input(&value).is_ok());
    }

    #[test]
    fn stop_contract_rejects_empty_or_excess_entries() {
        let mut value = request(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        }));
        value.stop = (0..64).map(|i| format!("s{i}")).collect();
        assert!(validate_input(&value).is_ok());

        value.stop.push("s64".into());
        assert_eq!(
            validate_input(&value).unwrap_err().status(),
            StatusCode::BAD_REQUEST
        );

        value.stop = vec![String::new()];
        assert_eq!(
            validate_input(&value).unwrap_err().status(),
            StatusCode::BAD_REQUEST
        );
    }
}

#[cfg(test)]
mod intervention_ledger_tests {
    use super::apply_failure_guards;

    fn request(messages: serde_json::Value) -> crate::openai::ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": messages,
            "report_interventions": true
        }))
        .expect("valid chat request fixture")
    }

    fn tool_call(id: &str, name: &str, arguments: &str) -> serde_json::Value {
        serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments}
            }]
        })
    }

    fn tool_result(id: &str, content: &str) -> serde_json::Value {
        serde_json::json!({"role": "tool", "tool_call_id": id, "content": content})
    }

    fn has(req: &crate::openai::ChatCompletionRequest, code: &str, action: &str) -> bool {
        req.intervention_ledger
            .events
            .iter()
            .any(|event| event.code == code && event.action == action && event.count > 0)
    }

    #[test]
    fn f32_f35_and_f52_mutations_are_recorded() {
        let mut req = request(serde_json::json!([
            {"role": "user", "content": "build it"},
            tool_call("c1", "Bash", r#"{"command":"cargo init"}"#),
            tool_result("c1", "[tool error]\ncargo: command not found"),
            {"role": "user", "content": "continue"},
            {"role": "user", "content": "please finish"}
        ]));

        apply_failure_guards(&mut req);

        assert!(has(&req, "F32", "duplicate_failed_tool_result"));
        assert!(has(&req, "F35", "trigger_failure_recovery"));
        assert!(has(&req, "F52", "prepend_failure_recovery_clause"));
    }

    #[test]
    fn f39_and_f29_mutations_are_recorded() {
        let mut req = request(serde_json::json!([
            {"role": "user", "content": "make a crate"},
            tool_call("c1", "Bash", r#"{"command":"cargo init"}"#),
            tool_result("c1", "[tool error]\ncargo: command not found"),
            tool_call("c2", "Bash", r#"{"command":"cargo init --offline"}"#),
            tool_result("c2", "[tool error]\ncargo: command not found"),
            tool_call("c3", "Bash", r#"{"command":"cargo init --name demo"}"#)
        ]));

        apply_failure_guards(&mut req);

        assert!(has(
            &req,
            "F39",
            "prepend_permanent_failure_circuit_breaker"
        ));
        assert!(has(&req, "F29", "inject_environment_facts"));
    }

    #[test]
    fn f49_and_successful_f50_mutations_are_recorded() {
        let write = r#"{"file_path":"/tmp/Cargo.toml","content":"x"}"#;
        let mut req = request(serde_json::json!([
            {"role": "user", "content": "repair it"},
            tool_call("c0", "Bash", r#"{"command":"cargo test"}"#),
            tool_result("c0", "[tool error]\nerror: missing src/main.rs"),
            tool_call("c1", "Write", write),
            tool_result("c1", "Wrote file successfully."),
            tool_call("c2", "Write", write)
        ]));

        apply_failure_guards(&mut req);

        assert!(has(&req, "F49", "prepend_duplicate_write_banner"));
        assert!(has(&req, "F50", "append_original_tool_error"));
    }

    #[test]
    fn f50_is_not_recorded_when_f49_has_no_original_error_to_append() {
        let write = r#"{"file_path":"/tmp/Cargo.toml","content":"x"}"#;
        let mut req = request(serde_json::json!([
            {"role": "user", "content": "write it"},
            tool_call("c1", "Write", write),
            tool_result("c1", "Wrote file successfully."),
            tool_call("c2", "Write", write)
        ]));

        apply_failure_guards(&mut req);

        assert!(has(&req, "F49", "prepend_duplicate_write_banner"));
        assert!(
            !req.intervention_ledger
                .events
                .iter()
                .any(|event| event.code == "F50")
        );
    }

    #[test]
    fn concurrent_system_notices_are_combined_without_overwriting() {
        let mut req = request(serde_json::json!([
            {"role": "user", "content": "make a crate"},
            tool_call("c1", "Bash", r#"{"command":"cargo init"}"#),
            tool_result("c1", "[tool error]\ncargo: command not found"),
            tool_call("c2", "Bash", r#"{"command":"cargo init --offline"}"#),
            tool_result("c2", "[tool error]\ncargo: command not found"),
            tool_call("c3", "Bash", r#"{"command":"cargo init --name demo"}"#),
            tool_result("c3", "[tool error]\ncargo: command not found")
        ]));

        apply_failure_guards(&mut req);

        assert_eq!(req.messages[0].role, "system");
        let sys = &req.messages[0].content.text;
        assert!(sys.contains("<flux_runtime_notice>"), "{sys}");
        assert!(sys.contains("<flux_circuit_breaker>"), "{sys}");
        assert!(sys.contains("<failure_recovery>"), "{sys}");
    }
}
