// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::fuzzy_match::fuzzy_match_tool_name;
use super::*;

/// Fix tool call arguments: schema-aware type coercion + backfill missing params.
///
/// The qwen3_coder XML format emits all parameter values as raw text. This function:
/// 1. **Type coercion**: Converts string values to the schema-expected type
///    (number, boolean, integer, object, array). Prevents "expected number,
///    received string" errors from clients like OpenCode.
/// 2. **Backfill**: fills a missing/empty required parameter only when a real
///    value can be derived from model-authored arguments or the tool definition.
///    It never invents a placeholder.
///
/// Matches vLLM's qwen3coder_tool_parser behavior (schema-aware type conversion).
///
/// Follow up to 8 local JSON-Pointer `$ref` hops (e.g. `#/$defs/Node`) against
/// `root`, returning the target sub-schema (or the last reachable object on
/// cycles / unresolved refs).
fn resolve_local_schema_ref<'a>(
    mut schema: &'a serde_json::Value,
    root: &'a serde_json::Value,
) -> &'a serde_json::Value {
    for _ in 0..8 {
        let Some(ref_str) = schema.get("$ref").and_then(|v| v.as_str()) else {
            break;
        };
        let Some(path) = ref_str.strip_prefix("#/") else {
            break;
        };
        let mut cur = root;
        let mut found = true;
        for seg in path.split('/') {
            let decoded = seg.replace("~1", "/").replace("~0", "~");
            let Some(next) = cur.get(&decoded) else {
                found = false;
                break;
            };
            cur = next;
        }
        if !found || std::ptr::eq(cur, schema) {
            break;
        }
        schema = cur;
    }
    schema
}

/// True when `schema` explicitly permits JSON `null` (`type: "null"`,
/// `type: [..., "null"]`, OpenAPI `nullable: true`, or an `anyOf`/`oneOf`
/// variant allowing `null`).
fn schema_allows_null(schema: &serde_json::Value, root: &serde_json::Value) -> bool {
    schema_allows_null_depth(schema, root, 0)
}

fn schema_allows_null_depth(
    schema: &serde_json::Value,
    root: &serde_json::Value,
    depth: usize,
) -> bool {
    if depth > 8 {
        return false;
    }
    if schema.get("nullable").and_then(|v| v.as_bool()) == Some(true) {
        return true;
    }
    let resolved = resolve_local_schema_ref(schema, root);
    if resolved.get("nullable").and_then(|v| v.as_bool()) == Some(true) {
        return true;
    }
    match resolved.get("type") {
        Some(serde_json::Value::String(s)) if s == "null" => return true,
        Some(serde_json::Value::Array(arr)) if arr.iter().any(|v| v.as_str() == Some("null")) => {
            return true;
        }
        _ => {}
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(variants) = resolved.get(key).and_then(|v| v.as_array())
            && variants
                .iter()
                .any(|v| schema_allows_null_depth(v, root, depth + 1))
        {
            return true;
        }
    }
    false
}

/// Resolves the effective non-null type from a JSON schema property, handling
/// local `$ref`, `"type": [...]` arrays, and `anyOf`/`oneOf` wrappers (e.g.
/// Pydantic v2's `Optional[int]` → `{"anyOf": [{"type":"integer"},{"type":"null"}]}`).
fn resolve_schema_type<'a>(
    schema: &'a serde_json::Value,
    root: &'a serde_json::Value,
) -> Option<&'a str> {
    resolve_schema_type_depth(schema, root, 0)
}

fn resolve_schema_type_depth<'a>(
    schema: &'a serde_json::Value,
    root: &'a serde_json::Value,
    depth: usize,
) -> Option<&'a str> {
    if depth > 8 {
        return None;
    }
    let resolved = resolve_local_schema_ref(schema, root);
    match resolved.get("type") {
        Some(serde_json::Value::String(t)) if t != "null" => return Some(t.as_str()),
        Some(serde_json::Value::Array(arr)) => {
            if let Some(t) = arr.iter().filter_map(|v| v.as_str()).find(|t| *t != "null") {
                return Some(t);
            }
        }
        _ => {}
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(variants) = resolved.get(key).and_then(|v| v.as_array()) {
            for variant in variants {
                if let Some(t) = resolve_schema_type_depth(variant, root, depth + 1) {
                    return Some(t);
                }
            }
        }
    }
    None
}

/// True when `schema` explicitly permits JSON type `ty` (via `"type": ty`,
/// `"type": [..., ty]`, or an `anyOf`/`oneOf` variant, following local `$ref`).
fn schema_allows_type(schema: &serde_json::Value, root: &serde_json::Value, ty: &str) -> bool {
    schema_allows_type_depth(schema, root, ty, 0)
}

fn schema_allows_type_depth(
    schema: &serde_json::Value,
    root: &serde_json::Value,
    ty: &str,
    depth: usize,
) -> bool {
    if depth > 8 {
        return false;
    }
    let resolved = resolve_local_schema_ref(schema, root);
    match resolved.get("type") {
        Some(serde_json::Value::String(t)) if t == ty => return true,
        Some(serde_json::Value::Array(arr)) if arr.iter().any(|v| v.as_str() == Some(ty)) => {
            return true;
        }
        _ => {}
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(variants) = resolved.get(key).and_then(|v| v.as_array())
            && variants
                .iter()
                .any(|v| schema_allows_type_depth(v, root, ty, depth + 1))
        {
            return true;
        }
    }
    false
}

/// Normalize a whole-number `f64` (e.g. `478.0`, `-5.0`) into an exact integer
/// `serde_json::Number`, rejecting fractional, non-finite, or out-of-range values.
fn whole_f64_to_int_number(f: f64) -> Option<serde_json::Number> {
    if !f.is_finite() || f.fract() != 0.0 {
        return None;
    }
    if (0.0..(u64::MAX as f64)).contains(&f) {
        Some(serde_json::Number::from(f as u64))
    } else if ((i64::MIN as f64)..0.0).contains(&f) {
        Some(serde_json::Number::from(f as i64))
    } else {
        None
    }
}

fn parse_agent_types(description: &str) -> Vec<String> {
    description
        .lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix("- ")?;
            let (name, _) = rest.split_once(':')?;
            let name = name.trim();
            (!name.is_empty()
                && name.len() <= 64
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'))
            .then(|| name.to_string())
        })
        .collect()
}

fn infer_default_subagent_type(description: Option<&str>) -> String {
    let candidates = description.map(parse_agent_types).unwrap_or_default();
    candidates
        .iter()
        .find(|candidate| candidate.to_ascii_lowercase().contains("general"))
        .or_else(|| candidates.first())
        .cloned()
        .unwrap_or_else(|| "general".to_string())
}

/// Derive a real value for a required parameter, or return `None` when the
/// server cannot know the model's intent. Paths, search strings, cities, and
/// other payload-bearing values must remain absent rather than becoming `""`.
fn derive_required_param(
    key: &str,
    args: &serde_json::Map<String, serde_json::Value>,
    tool_def: &ToolDefinition,
) -> Option<String> {
    match key {
        // Only derive `description` for shell/command tools where a real
        // `command` argument is present; never fabricate generic descriptions
        // for arbitrary tools like `create_ticket`.
        "description" => {
            let command = args
                .get("command")
                .and_then(|value| value.as_str())
                .filter(|cmd| !cmd.trim().is_empty())?;
            if command.chars().count() > 50 {
                let head: String = command.chars().take(47).collect();
                Some(format!("Run: {head}..."))
            } else {
                Some(format!("Run: {command}"))
            }
        }
        "subagent_type" | "subagentType" => Some(infer_default_subagent_type(
            tool_def.function.description.as_deref(),
        )),
        _ => None,
    }
}

/// Repair schema-coercible tool arguments and return the number of calls whose
/// published argument JSON changed.
pub fn backfill_required_params(calls: &mut [ToolCall], tools: &[ToolDefinition]) -> usize {
    let mut repaired_calls = 0usize;
    for call in calls.iter_mut() {
        let Some(tool_def) = tools.iter().find(|t| t.function.name == call.function.name) else {
            continue;
        };
        let Some(ref params_schema) = tool_def.function.parameters else {
            continue;
        };
        let resolved_params = resolve_local_schema_ref(params_schema, params_schema);
        let required = resolved_params
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();
        let properties = resolved_params
            .get("properties")
            .and_then(|p| p.as_object());
        let Ok(mut args) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
            &call.function.arguments,
        ) else {
            continue;
        };
        let mut changed = false;

        // 1. Normalize parameter names to match the schema first so step 2 can
        // coerce renamed keys (e.g. `maxResults` → `max_results`) to their
        // declared JSON Schema type.
        // The model sometimes emits camelCase (filePath) when the schema
        // defines snake_case (file_path), or vice versa. This is a known
        // Qwen3-Coder issue (vLLM #35347, llama.cpp #19382).
        if let Some(props) = properties {
            // Build case-insensitive lookup: "filepath" → "file_path" (schema name)
            let schema_normalized: std::collections::HashMap<String, &str> = props
                .keys()
                .map(|k| (k.to_lowercase().replace('_', ""), k.as_str()))
                .collect();

            let keys_to_fix: Vec<(String, String)> = args
                .keys()
                .filter(|k| !props.contains_key(*k))
                .filter_map(|k| {
                    let norm = k.to_lowercase().replace('_', "");
                    schema_normalized
                        .get(&norm)
                        .map(|schema_key| (k.clone(), schema_key.to_string()))
                })
                .collect();

            for (wrong_key, right_key) in keys_to_fix {
                if let Some(val) = args.remove(&wrong_key) {
                    args.entry(right_key).or_insert(val);
                    changed = true;
                }
            }
        }

        // 2. Coerce existing parameters to schema-expected types.
        if let Some(props) = properties {
            for (key, value) in args.iter_mut() {
                let Some(prop_schema) = props.get(key) else {
                    continue;
                };
                let expected = resolve_schema_type(prop_schema, params_schema);
                let coerced = match &*value {
                    serde_json::Value::String(s) => {
                        let trimmed = s.trim();
                        if trimmed == "null" && schema_allows_null(prop_schema, params_schema) {
                            Some(serde_json::Value::Null)
                        } else if let Some(expected) = expected {
                            match expected {
                                "number" => trimmed
                                    .parse::<f64>()
                                    .ok()
                                    .and_then(serde_json::Number::from_f64)
                                    .map(serde_json::Value::Number),
                                "integer" => trimmed
                                    .parse::<i64>()
                                    .ok()
                                    .map(|n| serde_json::Value::Number(n.into()))
                                    .or_else(|| {
                                        trimmed
                                            .parse::<u64>()
                                            .ok()
                                            .map(|n| serde_json::Value::Number(n.into()))
                                    })
                                    .or_else(|| {
                                        trimmed
                                            .parse::<f64>()
                                            .ok()
                                            .and_then(whole_f64_to_int_number)
                                            .map(serde_json::Value::Number)
                                    }),
                                "boolean" => match trimmed.to_lowercase().as_str() {
                                    "true" | "1" | "yes" => Some(serde_json::Value::Bool(true)),
                                    "false" | "0" | "no" => Some(serde_json::Value::Bool(false)),
                                    _ => None,
                                },
                                "object" | "array" => {
                                    serde_json::from_str::<serde_json::Value>(trimmed)
                                        .ok()
                                        .filter(|v| {
                                            matches!(
                                                (expected, v),
                                                ("object", serde_json::Value::Object(_))
                                                    | ("array", serde_json::Value::Array(_))
                                            )
                                        })
                                }
                                _ => None, // "string" or unknown — keep as-is
                            }
                        } else {
                            None
                        }
                    }
                    serde_json::Value::Number(n) => {
                        if !n.is_i64()
                            && !n.is_u64()
                            && schema_allows_type(prop_schema, params_schema, "integer")
                            && !schema_allows_type(prop_schema, params_schema, "number")
                        {
                            n.as_f64()
                                .and_then(whole_f64_to_int_number)
                                .map(serde_json::Value::Number)
                        } else if expected == Some("string")
                            && !schema_allows_type(prop_schema, params_schema, "number")
                            && !schema_allows_type(prop_schema, params_schema, "integer")
                        {
                            Some(serde_json::Value::String(n.to_string()))
                        } else {
                            None
                        }
                    }
                    serde_json::Value::Bool(b) => {
                        if expected == Some("string")
                            && !schema_allows_type(prop_schema, params_schema, "boolean")
                        {
                            Some(serde_json::Value::String(b.to_string()))
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some(new_val) = coerced {
                    *value = new_val;
                    changed = true;
                }
            }
        }

        // 3. Derive only values the server can determine honestly. An absent
        // key makes the client's JSON-Schema `required` check fail loudly; a
        // fabricated empty string is valid `type:string` and can execute the
        // wrong call without any downstream layer knowing Atlas authored it.
        for key in &required {
            let model_supplied = match args.get(*key) {
                Some(serde_json::Value::String(value)) => !value.trim().is_empty(),
                Some(_) => true,
                None => false,
            };
            if model_supplied {
                continue;
            }
            if let Some(derived) = derive_required_param(key, &args, tool_def) {
                args.insert(key.to_string(), serde_json::Value::String(derived));
                changed = true;
            }
        }

        if changed
            && let Ok(new_args) = serde_json::to_string(&serde_json::Value::Object(args))
            && new_args != call.function.arguments
        {
            call.function.arguments = new_args;
            repaired_calls = repaired_calls.saturating_add(1);
        }
    }
    repaired_calls
}

/// Check if a tool call has empty required parameters that can't be auto-filled.
/// Returns the names of empty required params, or empty vec if all are filled.
pub fn find_empty_required_params(call: &ToolCall, tools: &[ToolDefinition]) -> Vec<String> {
    let Some(tool_def) = tools.iter().find(|t| t.function.name == call.function.name) else {
        return Vec::new();
    };
    let Some(ref params_schema) = tool_def.function.parameters else {
        return Vec::new();
    };
    let resolved_params = resolve_local_schema_ref(params_schema, params_schema);
    let required = resolved_params
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let properties = resolved_params
        .get("properties")
        .and_then(|p| p.as_object());
    let args: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&call.function.arguments).unwrap_or_default();
    let mut empty = Vec::new();
    for key in &required {
        match args.get(key.as_str()) {
            None => empty.push(key.clone()),
            Some(serde_json::Value::String(s)) if s.trim().is_empty() => empty.push(key.clone()),
            Some(serde_json::Value::Null) => {
                let allows_null = properties
                    .and_then(|props| props.get(key.as_str()))
                    .is_some_and(|prop| schema_allows_null(prop, params_schema));
                if !allows_null {
                    empty.push(key.clone());
                }
            }
            _ => {}
        }
    }
    empty
}

/// Normalize file paths in tool call arguments to be relative to the working directory.
///
/// OPENCODE BUG FIX (2026-04-22): the previous behaviour stripped the leading `/`
/// of any absolute path NOT under cwd, mangling user-intended paths like
/// `/tmp/calc-test16/calc.py` into `tmp/calc-test16/calc.py`. opencode then
/// resolved that relative path under `Instance.directory` (= `$HOME`), so the
/// file ended up at `$HOME/tmp/calc-test16/calc.py` instead of
/// `/tmp/calc-test16/`. The model spent 8+ turns trying to "fix" the directory
/// before the user noticed.
///
/// New behaviour:
/// - Paths under cwd → made relative (still helpful for Claude-Code-style clients)
/// - Paths starting with `/` but NOT under cwd → **PASS THROUGH UNCHANGED**.
///   The model knew what it wanted (e.g. user said "put it in /tmp/..."); we
///   should not second-guess. If it really is wrong, the filesystem op will
///   fail with a clear error and the model can self-correct.
/// - Already relative paths → unchanged
/// Normalize paths under `cwd` and return the number of calls whose published
/// argument JSON changed.
pub fn normalize_paths(calls: &mut [ToolCall], cwd: &str) -> usize {
    // Common parameter names that contain file paths
    const PATH_KEYS: &[&str] = &["file_path", "filePath", "path", "file"];
    let cwd_slash = if cwd.ends_with('/') {
        cwd.to_string()
    } else {
        format!("{cwd}/")
    };

    let mut repaired_calls = 0usize;
    for call in calls.iter_mut() {
        let Ok(mut args) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
            &call.function.arguments,
        ) else {
            continue;
        };
        let mut changed = false;
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key) {
                if !path.starts_with('/') {
                    continue; // Already relative — leave it
                }
                if !path.starts_with(&cwd_slash) {
                    // Absolute path NOT under cwd — pass through verbatim. The
                    // user explicitly asked for this location (e.g. "/tmp/..."),
                    // and trimming `/` here breaks downstream clients that
                    // resolve relative paths against THEIR own working dir.
                    continue;
                }
                let new_path = path[cwd_slash.len()..].to_string();
                if new_path != *path && !new_path.is_empty() {
                    args.insert(key.to_string(), serde_json::Value::String(new_path));
                    changed = true;
                }
            }
        }
        if changed
            && let Ok(new_args) = serde_json::to_string(&serde_json::Value::Object(args))
            && new_args != call.function.arguments
        {
            call.function.arguments = new_args;
            repaired_calls = repaired_calls.saturating_add(1);
        }
    }
    repaired_calls
}

// ── Tool call validation ──

const FILE_TOOLS: &[&str] = &["Write", "write", "Edit", "edit", "Read", "read"];
const PATH_KEYS: &[&str] = &["file_path", "filePath", "path"];
const WRITE_FAMILY: &[&str] = &[
    "Write",
    "write",
    "Edit",
    "edit",
    "MultiEdit",
    "multiEdit",
    "multi_edit",
    "write_file",
    "writeFile",
];
const SHELL_FAMILY: &[&str] = &[
    "bash", "Bash", "shell", "Shell", "exec", "Exec", "run", "Run", "execute", "Execute",
    "terminal", "Terminal",
];
const CMD_KEYS: &[&str] = &["command", "cmd", "script", "code"];

/// Result of validating a batch of tool calls against their schemas.
pub struct ValidatedToolCalls {
    /// Tool calls that passed all validations.
    pub valid: Vec<ToolCall>,
    /// Human-readable error messages for invalid calls.
    /// These should be injected into the response content so the model
    /// sees clear, actionable feedback instead of cryptic client errors.
    pub errors: Vec<String>,
}

/// Validation severity determines whether a parsed call can be attached to
/// the response. Missing ordinary/read-only arguments remain attached so the
/// client can return actionable schema feedback; unexecutable mutation/shell
/// calls and structurally invalid calls are withheld.
#[derive(Debug)]
pub enum ToolCallIssue {
    MissingParam(String),
    EmptyRequired(String),
    Hard(String),
}

impl ToolCallIssue {
    pub fn message(&self) -> &str {
        match self {
            Self::MissingParam(message) | Self::EmptyRequired(message) | Self::Hard(message) => {
                message
            }
        }
    }

    pub fn into_message(self) -> String {
        match self {
            Self::MissingParam(message) | Self::EmptyRequired(message) | Self::Hard(message) => {
                message
            }
        }
    }
}

/// Validate tool calls against their schemas. Returns valid calls and
/// error messages for invalid ones.
///
/// Checks:
/// 1. Tool name exists in definitions
/// 2. Arguments are valid JSON
/// 3. Required params are present, with family-specific attachment policy
/// 4. file_path params don't look like directories (end with `/`)
pub fn validate_tool_calls(
    mut calls: Vec<ToolCall>,
    tools: &[ToolDefinition],
) -> ValidatedToolCalls {
    let mut valid = Vec::new();
    let mut errors = Vec::new();

    for call in &mut calls {
        // Fuzzy name repair: if model hallucinates a close-but-wrong name,
        // map to the closest available tool (NVFP4 models often drop prefixes
        // like "get_" or use abbreviations like "weather" for "get_weather").
        if tools.iter().all(|t| t.function.name != call.function.name)
            && let Some(best) = fuzzy_match_tool_name(&call.function.name, tools)
        {
            tracing::info!(
                "Fuzzy tool name repair: '{}' -> '{}'",
                call.function.name,
                best
            );
            call.function.name = best;
        }
        match assess_tool_call(call, tools) {
            Ok(()) => valid.push(call.clone()),
            Err(ToolCallIssue::MissingParam(message)) => {
                valid.push(call.clone());
                errors.push(message);
            }
            Err(issue) => errors.push(issue.into_message()),
        }
    }

    ValidatedToolCalls { valid, errors }
}

/// Compatibility wrapper for callers that only need pass/fail.
pub fn validate_single_tool_call(call: &ToolCall, tools: &[ToolDefinition]) -> Result<(), String> {
    assess_tool_call(call, tools).map_err(ToolCallIssue::into_message)
}

pub fn assess_tool_call(call: &ToolCall, tools: &[ToolDefinition]) -> Result<(), ToolCallIssue> {
    let name = &call.function.name;

    // 1. Check tool name exists
    let Some(tool_def) = tools.iter().find(|t| t.function.name == *name) else {
        let available: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        return Err(ToolCallIssue::Hard(format!(
            "Error: Unknown tool '{}'. Available tools: {}",
            name,
            available.join(", ")
        )));
    };

    // 2. Check arguments are valid JSON
    let args: serde_json::Map<String, serde_json::Value> =
        match serde_json::from_str(&call.function.arguments) {
            Ok(a) => a,
            Err(_) => {
                let preview: String = call.function.arguments.chars().take(100).collect();
                return Err(ToolCallIssue::Hard(format!(
                    "Error: {name} arguments must be valid JSON. Got: {preview}"
                )));
            }
        };

    // 3. Check required params are present. Do NOT enforce non-empty strings —
    // that is the client's schema concern. Empty-string rejection here broke
    // Theia IDE's getWorkspaceFileList, which legitimately passes path="".
    if let Some(ref params_schema) = tool_def.function.parameters {
        let resolved_params = resolve_local_schema_ref(params_schema, params_schema);
        let required: Vec<&str> = resolved_params
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        let missing: Vec<&str> = required
            .iter()
            .copied()
            .filter(|key| args.get(*key).is_none())
            .collect();
        if !missing.is_empty() {
            // Do not let JSON-Schema array ordering weaken the disposition.
            // If a write has neither content nor path and `content` appears
            // first, the missing path must still withhold the call; likewise
            // for a shell schema that lists metadata before its command.
            let unexecutable = missing.iter().find(|key| {
                (WRITE_FAMILY.contains(&name.as_str()) && PATH_KEYS.contains(key))
                    || (SHELL_FAMILY.contains(&name.as_str()) && CMD_KEYS.contains(key))
            });
            let key = unexecutable.copied().unwrap_or(missing[0]);
            let message =
                format!("Error: {name} requires parameter '{key}' but it was not provided.");
            return Err(if unexecutable.is_some() {
                ToolCallIssue::EmptyRequired(message)
            } else {
                ToolCallIssue::MissingParam(message)
            });
        }
    }

    // 4. Path-specific validation for file tools.
    // File mutation tools must have a non-empty path: when a model
    // self-truncates content and grammar-completes `filePath` with `""`,
    // downstream write tools fail with EISDIR and can loop on retries.
    // Rejecting here turns the malformed tool_call into an issue so the
    // response falls through cleanly. Read/Glob/LS keep lenient behavior
    // (some IDE clients legitimately pass `path=""` to list the workspace root).
    if WRITE_FAMILY.contains(&name.as_str()) {
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key)
                && path.trim().is_empty()
            {
                return Err(ToolCallIssue::EmptyRequired(format!(
                    "Error: {name} requires a non-empty '{key}'. \
                         Got empty string — provide an absolute path \
                         like '/path/to/file'."
                )));
            }
        }
    }
    if SHELL_FAMILY.contains(&name.as_str()) {
        for key in CMD_KEYS {
            if let Some(serde_json::Value::String(command)) = args.get(*key)
                && command.trim().len() < 2
            {
                return Err(ToolCallIssue::EmptyRequired(format!(
                    "Error: {name} requires a non-empty '{key}'. Provide the shell command to execute."
                )));
            }
        }
    }
    if FILE_TOOLS.contains(&name.as_str()) {
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key) {
                if path.ends_with('/') {
                    return Err(ToolCallIssue::Hard(format!(
                        "Error: {} file_path must be a FILE, not a directory. Got '{}'. Use e.g. '{}/index.ts'",
                        name,
                        path,
                        path.trim_end_matches('/')
                    )));
                }
                // Check if it looks like just a directory name (no extension, no dots, no uppercase).
                // Only reject under opt-in agent guards — valid POSIX files like `hosts`,
                // `justfile`, `notes`, or `passwd` are lowercase and extensionless.
                if crate::api::chat_phases::agent_guards_enabled()
                    && !path.is_empty()
                    && !path.contains('.')
                    && !path.contains('/')
                    && path
                        .chars()
                        .all(|c| c.is_lowercase() || c == '-' || c == '_')
                {
                    return Err(ToolCallIssue::Hard(format!(
                        "Error: {} file_path '{}' looks like a directory. Add a filename, e.g. '{}/index.ts'",
                        name, path, path
                    )));
                }
            }
        }
    }

    Ok(())
}
