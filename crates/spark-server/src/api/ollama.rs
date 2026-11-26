// SPDX-License-Identifier: AGPL-3.0-only

//! Zero-config Ollama discovery endpoints (`GET /api/tags`, `POST /api/show`).
//!
//! Exposes the currently loaded Atlas model in Ollama's exact wire schema so
//! OpenCode, Cursor, Continue, Cline, and OpenWebUI auto-discover model name,
//! context window, quantization, and tool/thinking capabilities without manual
//! JSON configuration.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};

use crate::AppState;
use crate::main_modules::model_host::ModelHost;

/// Fixed RFC-3339 timestamp reported for deterministic Ollama tag metadata.
const DEFAULT_MODIFIED_AT: &str = "2026-01-01T00:00:00Z";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct OllamaModelDetails {
    pub parent_model: String,
    pub format: String,
    pub family: String,
    pub families: Vec<String>,
    pub parameter_size: String,
    pub quantization_level: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct OllamaTagEntry {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: OllamaModelDetails,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaTagEntry>,
}

#[derive(Debug, Default, Deserialize)]
pub struct OllamaShowRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

/// Infer Ollama `family` from the served model identifier.
#[must_use]
pub fn infer_model_family(model_name: &str, dsv41: bool) -> String {
    if dsv41 {
        return "deepseek".to_string();
    }
    let lower = model_name.to_ascii_lowercase();
    for (needle, family) in [
        ("qwen3", "qwen3"),
        ("qwen", "qwen"),
        ("deepseek", "deepseek"),
        ("glm", "glm"),
        ("nemotron", "nemotron"),
        ("minimax", "minimax"),
        ("gemma", "gemma"),
        ("mistral", "mistral"),
        ("llama", "llama"),
    ] {
        if lower.contains(needle) {
            return family.to_string();
        }
    }
    "transformer".to_string()
}

/// Infer human-readable parameter count (e.g. `"80B"`, `"35B"`) from model ID.
#[must_use]
pub fn infer_parameter_size(model_name: &str) -> String {
    for raw_seg in model_name.split(['/', '-', '_', ' ']) {
        let seg = raw_seg.trim();
        let Some((num_part, unit)) = seg
            .strip_suffix('b')
            .map(|n| (n, "B"))
            .or_else(|| seg.strip_suffix('B').map(|n| (n, "B")))
            .or_else(|| seg.strip_suffix('m').map(|n| (n, "M")))
            .or_else(|| seg.strip_suffix('M').map(|n| (n, "M")))
        else {
            continue;
        };
        if num_part.chars().next().is_some_and(|c| c.is_ascii_digit())
            && num_part.chars().all(|c| c.is_ascii_digit() || c == '.')
            && num_part.matches('.').count() <= 1
            && !num_part.ends_with('.')
        {
            return format!("{num_part}{unit}");
        }
    }
    "unknown".to_string()
}

/// Infer quantization label (e.g. `"NVFP4"`, `"FP8"`, `"BF16"`) from model ID.
#[must_use]
pub fn infer_quantization_level(model_name: &str) -> String {
    let lower = model_name.to_ascii_lowercase();
    for (needle, quant) in [
        ("nvfp4", "NVFP4"),
        ("mxfp4", "MXFP4"),
        ("fp4", "NVFP4"),
        ("fp8", "FP8"),
        ("awq", "AWQ"),
        ("gptq", "GPTQ"),
        ("int4", "INT4"),
        ("int8", "INT8"),
        ("bf16", "BF16"),
        ("fp16", "FP16"),
    ] {
        if lower.contains(needle) {
            return quant.to_string();
        }
    }
    "BF16".to_string()
}

#[must_use]
pub fn model_digest_hex(model_name: &str) -> String {
    let digest = ring::digest::digest(&ring::digest::SHA256, model_name.as_bytes());
    let mut out = String::with_capacity(64);
    for b in digest.as_ref() {
        use std::fmt::Write;
        let _ = write!(out, "{b:02x}");
    }
    out
}

#[must_use]
pub fn build_model_details(model_name: &str, dsv41: bool, has_vision: bool) -> OllamaModelDetails {
    let family = infer_model_family(model_name, dsv41);
    let mut families = vec![family.clone()];
    if has_vision {
        families.push("clip".to_string());
    }
    OllamaModelDetails {
        parent_model: String::new(),
        format: "safetensors".to_string(),
        family,
        families,
        parameter_size: infer_parameter_size(model_name),
        quantization_level: infer_quantization_level(model_name),
    }
}

#[must_use]
pub fn build_capabilities(
    has_tools: bool,
    supports_thinking: bool,
    disable_thinking: bool,
    has_vision: bool,
) -> Vec<String> {
    let mut caps = vec!["completion".to_string()];
    if has_tools {
        caps.push("tools".to_string());
    }
    if supports_thinking && !disable_thinking {
        caps.push("thinking".to_string());
    }
    if has_vision {
        caps.push("vision".to_string());
    }
    caps
}

/// Match an incoming Ollama model selector (`"model"`, `"model:latest"`, or
/// trailing HF basename) against the loaded `served_name`.
#[must_use]
pub fn matches_loaded_model(requested: Option<&str>, served_name: &str) -> bool {
    let Some(raw) = requested.map(str::trim).filter(|s| !s.is_empty()) else {
        return true;
    };
    let normalized = raw.strip_suffix(":latest").unwrap_or(raw);
    let served_norm = served_name.strip_suffix(":latest").unwrap_or(served_name);
    if normalized.eq_ignore_ascii_case(served_norm) {
        return true;
    }
    if let Some((_, basename)) = served_norm.rsplit_once('/')
        && normalized.eq_ignore_ascii_case(basename)
    {
        return true;
    }
    false
}

fn estimated_allocated_bytes() -> u64 {
    match (
        crate::tui::data::gpu_free_bytes(),
        spark_runtime::gpu::baseline_free_bytes(),
    ) {
        (Some(free), Some(baseline)) if baseline > free => (baseline - free) as u64,
        _ => 0,
    }
}

fn build_tag_entry(state: &AppState) -> OllamaTagEntry {
    let has_vision = state.vision_config.is_some() || state.dsv41_vision.is_some();
    OllamaTagEntry {
        name: state.model_name.clone(),
        model: state.model_name.clone(),
        modified_at: DEFAULT_MODIFIED_AT.to_string(),
        size: estimated_allocated_bytes(),
        digest: model_digest_hex(&state.model_name),
        details: build_model_details(&state.model_name, state.dsv41, has_vision),
    }
}

/// `GET /api/tags` — Ollama model list discovery.
pub async fn ollama_tags(State(host): State<Arc<ModelHost>>) -> Response {
    let models = match host.current() {
        Some(state) => vec![build_tag_entry(&state)],
        None => Vec::new(),
    };
    Json(OllamaTagsResponse { models }).into_response()
}

/// `POST /api/show` — Ollama model detail & capability inspection.
pub async fn ollama_show(State(host): State<Arc<ModelHost>>, body: axum::body::Bytes) -> Response {
    let req: OllamaShowRequest = if body.is_empty() {
        OllamaShowRequest::default()
    } else {
        match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": format!("invalid json: {e}") })),
                )
                    .into_response();
            }
        }
    };

    let Some(state) = host.current() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": "no model is currently loaded" })),
        )
            .into_response();
    };

    let requested = req.model.as_deref().or(req.name.as_deref());
    if !matches_loaded_model(requested, &state.model_name) {
        let missing = requested.unwrap_or_default();
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("model '{missing}' not found") })),
        )
            .into_response();
    }

    let has_vision = state.vision_config.is_some() || state.dsv41_vision.is_some();
    let details = build_model_details(&state.model_name, state.dsv41, has_vision);
    let capabilities = build_capabilities(
        state.tool_call_parser.is_some() || state.dsv41,
        state.tokenizer.supports_thinking() || state.reasoning_parser.is_some() || state.dsv41,
        state.disable_thinking,
        has_vision,
    );

    let arch_ctx_key = format!("{}.context_length", details.family);
    let mut model_info = serde_json::Map::new();
    model_info.insert(
        "general.architecture".to_string(),
        serde_json::Value::String(details.family.clone()),
    );
    model_info.insert(
        "general.basename".to_string(),
        serde_json::Value::String(state.model_name.clone()),
    );
    model_info.insert(
        "general.parameter_count".to_string(),
        serde_json::Value::String(details.parameter_size.clone()),
    );
    model_info.insert(
        "general.quantization_version".to_string(),
        serde_json::Value::String(details.quantization_level.clone()),
    );
    model_info.insert(
        "context_length".to_string(),
        serde_json::Value::from(state.max_seq_len),
    );
    model_info.insert(arch_ctx_key, serde_json::Value::from(state.max_seq_len));

    let payload = serde_json::json!({
        "modelfile": format!("FROM {}", state.model_name),
        "parameters": format!(
            "temperature {}\ntop_k {}\ntop_p {}",
            state.default_temperature,
            state.default_top_k,
            state.default_top_p,
        ),
        "template": "{{ .Prompt }}",
        "details": details,
        "model_info": model_info,
        "capabilities": capabilities,
    });
    Json(payload).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_family_size_and_quantization_for_atlas_checkpoints() {
        let d = build_model_details("nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4", false, false);
        assert_eq!(d.family, "qwen3");
        assert_eq!(d.parameter_size, "80B");
        assert_eq!(d.quantization_level, "NVFP4");

        let ds = build_model_details("deepseek-ai/DeepSeek-V4.1-FP8", true, true);
        assert_eq!(ds.family, "deepseek");
        assert_eq!(ds.quantization_level, "FP8");
        assert_eq!(ds.families, vec!["deepseek", "clip"]);

        assert_eq!(infer_parameter_size("org/modèle-8b"), "8B");
        assert_eq!(infer_parameter_size("ümläüt_モデル-350m"), "350M");
        assert_eq!(
            infer_quantization_level("openai/gpt-oss-120b-MXFP4"),
            "MXFP4"
        );
        assert_eq!(infer_quantization_level("nvidia/model-FP4"), "NVFP4");
    }

    #[test]
    fn matches_loaded_model_handles_latest_suffix_and_basename() {
        let served = "nvidia/Qwen3.8-Flash-Next-NVFP4";
        assert!(matches_loaded_model(None, served));
        assert!(matches_loaded_model(Some(""), served));
        assert!(matches_loaded_model(Some(served), served));
        assert!(matches_loaded_model(
            Some("nvidia/Qwen3.8-Flash-Next-NVFP4:latest"),
            served
        ));
        assert!(matches_loaded_model(
            Some("Qwen3.8-Flash-Next-NVFP4:latest"),
            served
        ));
        assert!(!matches_loaded_model(Some("llama3:8b"), served));
    }

    #[test]
    fn capabilities_reflect_tools_thinking_and_kill_switch() {
        assert_eq!(
            build_capabilities(true, true, false, false),
            vec!["completion", "tools", "thinking"]
        );
        assert_eq!(
            build_capabilities(true, true, true, true),
            vec!["completion", "tools", "vision"]
        );
    }
}
