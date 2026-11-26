// SPDX-License-Identifier: AGPL-3.0-only

//! Register as a child of serve_phases::runtime. Resolves the actual parser
//! without constructing a model, scheduler, tokenizer or GPU backend.
//! glm5_next auto-selects `glm_xml` through tool_defaults.toml; its MODEL.toml
//! carries no `[behavior].tool_call_parser` override (read here as empty).

use super::resolve_tool_call_parser;
use atlas_core::{config::ModelConfig, target::KernelTarget};
use atlas_kernels::{ModelBehavior, SamplingPresets, TargetPtxSet};
use clap::Parser;

fn config() -> ModelConfig {
    // Only model_type is consumed by the production resolver; no geometry is
    // asserted by this metadata-only fixture and no weights are admitted.
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config
}

fn target() -> TargetPtxSet {
    static METADATA: std::sync::LazyLock<toml::Value> = std::sync::LazyLock::new(|| {
        toml::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/gb10/glm5.3-flash/MODEL.toml"
        )))
        .unwrap()
    });
    let behavior = ModelBehavior {
        tool_call_parser: METADATA
            .get("behavior")
            .and_then(|v| v.get("tool_call_parser"))
            .and_then(|v| v.as_str())
            .unwrap_or(""),
        ..ModelBehavior::default()
    };
    TargetPtxSet {
        target: KernelTarget {
            arch: "sm_121",
            model: "glm5.3-flash",
            quant: "exl3",
        },
        modules: vec![],
        sampling: SamplingPresets::default(),
        behavior,
        model_type_matches: vec![],
        dflash: None,
    }
}

fn qwen4_config() -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "qwen4_exp".into();
    config
}

fn qwen4_target() -> TargetPtxSet {
    static METADATA: std::sync::LazyLock<toml::Value> = std::sync::LazyLock::new(|| {
        toml::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/gb10/qwen3.8-flash-next/MODEL.toml"
        )))
        .unwrap()
    });
    let mut target = target();
    target.target.model = "qwen3.8-flash-next";
    target.behavior.tool_call_parser = METADATA
        .get("behavior")
        .and_then(|v| v.get("tool_call_parser"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    target.behavior.thinking_default = METADATA
        .get("behavior")
        .and_then(|v| v.get("thinking_default"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    target.behavior.thinking_in_tools = METADATA
        .get("behavior")
        .and_then(|v| v.get("thinking_in_tools"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    target.behavior.honor_eos_inside_thinking = METADATA
        .get("behavior")
        .and_then(|v| v.get("honor_eos_inside_thinking"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    target.behavior.max_thinking_budget = METADATA
        .get("behavior")
        .and_then(|v| v.get("max_thinking_budget"))
        .and_then(|v| v.as_integer())
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or_default();
    target
}

#[test]
fn glm_tool_contract_registry_resolves_actual_default_native_parser() {
    let args = crate::cli::ServeArgs::try_parse_from(["spark", "fixture"]).unwrap();
    let parser = resolve_tool_call_parser(&args, &target(), &config())
        .unwrap()
        .expect("glm5_next must auto-select its native tool parser");
    assert_eq!(
        parser.name(),
        "glm_xml",
        "do not alias an incompatible wire grammar"
    );
    let tools: Vec<crate::tool_parser::ToolDefinition> =
        serde_json::from_value(serde_json::json!([{"type":"function", "function":{
        "name":"inspect", "parameters":{"type":"object", "properties":{
            "literal":{"type":"string"}}}}}]))
        .unwrap();
    let prompt = parser.system_prompt(&tools, &crate::tool_parser::ToolChoice::Mode("auto".into()));
    assert!(prompt.contains("<arg_key>"));
    assert!(prompt.contains("<arg_value>"));
    assert!(!prompt.contains("<function="));
    assert!(!prompt.contains("<invoke name="));
}

#[test]
fn glm_tool_contract_native_cli_name_resolves_and_roundtrips() {
    let format: crate::tool_parser::ToolCallFormat = "glm_xml".parse().unwrap();
    assert_eq!(format.name(), "glm_xml");
    let args = crate::cli::ServeArgs::try_parse_from([
        "spark",
        "fixture",
        "--tool-call-parser",
        "glm_xml",
    ])
    .unwrap();
    assert_eq!(
        resolve_tool_call_parser(&args, &target(), &config())
            .unwrap()
            .unwrap()
            .name(),
        "glm_xml"
    );
}

#[test]
fn glm_tool_contract_existing_explicit_override_precedence_is_preserved() {
    let args =
        crate::cli::ServeArgs::try_parse_from(["spark", "fixture", "--tool-call-parser", "hermes"])
            .unwrap();
    let mut target = target();
    target.behavior.tool_call_parser = "qwen3_coder";
    assert_eq!(
        resolve_tool_call_parser(&args, &target, &config())
            .unwrap()
            .unwrap()
            .name(),
        "hermes"
    );
    let args = crate::cli::ServeArgs::try_parse_from(["spark", "fixture"]).unwrap();
    assert_eq!(
        resolve_tool_call_parser(&args, &target, &config())
            .unwrap()
            .unwrap()
            .name(),
        "qwen3_coder"
    );
}

#[test]
fn glm_tool_contract_invalid_override_is_error_not_disabled_success() {
    let args = crate::cli::ServeArgs::try_parse_from([
        "spark",
        "fixture",
        "--tool-call-parser",
        "unknown_glm_parser",
    ])
    .unwrap();
    assert!(resolve_tool_call_parser(&args, &target(), &config()).is_err());
}

#[test]
fn qwen4_flash_next_resolves_native_xml_parser_from_target_metadata() {
    let args = crate::cli::ServeArgs::try_parse_from(["spark", "fixture"]).unwrap();
    let parser = resolve_tool_call_parser(&args, &qwen4_target(), &qwen4_config())
        .unwrap()
        .expect("qwen4_exp must select its native XML parser");
    assert_eq!(parser.name(), "qwen3_coder");

    let target = qwen4_target();
    assert!(target.behavior.thinking_default);
    assert!(target.behavior.thinking_in_tools);
    assert!(target.behavior.honor_eos_inside_thinking);
    assert_eq!(target.behavior.max_thinking_budget, 262144);
}

#[test]
fn qwen4_flash_next_registry_default_and_explicit_hermes_precedence() {
    let args = crate::cli::ServeArgs::try_parse_from(["spark", "fixture"]).unwrap();
    let mut target = qwen4_target();
    target.behavior.tool_call_parser = "";
    assert_eq!(
        resolve_tool_call_parser(&args, &target, &qwen4_config())
            .unwrap()
            .unwrap()
            .name(),
        "qwen3_coder",
        "tool_defaults.toml must cover direct qwen4_exp launches"
    );

    let explicit =
        crate::cli::ServeArgs::try_parse_from(["spark", "fixture", "--tool-call-parser", "hermes"])
            .unwrap();
    assert_eq!(
        resolve_tool_call_parser(&explicit, &qwen4_target(), &qwen4_config())
            .unwrap()
            .unwrap()
            .name(),
        "hermes",
        "an operator override must still win"
    );
}
