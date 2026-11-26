// SPDX-License-Identifier: AGPL-3.0-only

//! Model-behavior settings affecting decode output: the content-loop watchdog
//! is off for Qwen3.8-27B and Flash-Next (`[behavior].enable_loop_watchdog`
//! defaults to false in build_parse.rs, so an absent key means off), and
//! Gemma-4-31B's sampling presets carry a neutral repetition penalty.

use std::path::PathBuf;

fn model_toml(model: &str) -> toml::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10")
        .join(model)
        .join("MODEL.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    toml::from_str(&text).unwrap()
}

fn loop_watchdog(model: &str) -> Option<bool> {
    model_toml(model)
        .get("behavior")
        .and_then(|b| b.get("enable_loop_watchdog"))
        .map(|v| v.as_bool().expect("enable_loop_watchdog must be a bool"))
}

#[test]
fn qwen38_27b_runs_without_the_content_loop_watchdog() {
    // It truncated ordinary lists / repeated guard clauses as finish_reason=length.
    assert_eq!(loop_watchdog("qwen3.8-27b"), Some(false));
}

#[test]
fn flash_next_keeps_the_default_off_setting() {
    assert_eq!(loop_watchdog("qwen3.8-flash-next"), None);
}

#[test]
fn compiled_targets_agree_with_model_toml() {
    // Meaningful in whichever target set this test binary was built for.
    for target in atlas_kernels::available_targets() {
        let model = target.target.model;
        if model == "qwen3.8-27b" || model == "qwen3.8-flash-next" {
            assert!(
                !target.behavior.enable_loop_watchdog,
                "{model} compiled with the watchdog on"
            );
            assert!(
                !target.behavior.confidence_early_stop,
                "{model} compiled with confidence early-stop on"
            );
        }
    }
}

#[test]
fn gemma_4_31b_presets_use_a_neutral_repetition_penalty() {
    let value = model_toml("gemma-4-31b");
    let sampling = value
        .get("sampling")
        .and_then(|s| s.as_table())
        .expect("[sampling]");
    for (category, table) in sampling {
        let penalty = table
            .get("repetition_penalty")
            .map(|v| v.as_float().unwrap())
            .unwrap_or(1.0);
        assert_eq!(
            penalty, 1.0,
            "gemma-4-31b [sampling.{category}] repetition_penalty"
        );
    }
    // The compiled table agrees, in whichever target set this test binary was built for.
    for target in atlas_kernels::available_targets() {
        if target.target.model == "gemma-4-31b" {
            let s = &target.sampling;
            for (name, category) in [
                ("thinking_text", &s.thinking_text),
                ("thinking_coding", &s.thinking_coding),
                ("non_thinking", &s.non_thinking),
                ("tools", &s.tools),
            ] {
                assert_eq!(
                    category.repetition_penalty, 1.0,
                    "compiled gemma-4-31b {name}"
                );
            }
        }
    }
}
