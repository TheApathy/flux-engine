// SPDX-License-Identifier: AGPL-3.0-only

//! Flash-Next thinking defaults: thinking requests sample at the checkpoint's
//! generation_config.json, while the thinking-off presets stay greedy for the
//! release identity gates.

use std::path::PathBuf;

fn model_toml(model: &str) -> toml::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10")
        .join(model)
        .join("MODEL.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    toml::from_str(&text).unwrap()
}

fn preset(value: &toml::Value, category: &str) -> (f64, f64, i64, f64) {
    let table = value
        .get("sampling")
        .and_then(|s| s.get(category))
        .unwrap_or_else(|| panic!("[sampling.{category}]"));
    let float = |key: &str| table.get(key).and_then(|v| v.as_float()).unwrap();
    (
        float("temperature"),
        float("top_p"),
        table.get("top_k").and_then(|v| v.as_integer()).unwrap(),
        float("presence_penalty"),
    )
}

#[test]
fn flash_next_thinking_presets_match_generation_config() {
    let value = model_toml("qwen3.8-flash-next");
    for category in ["thinking_text", "thinking_coding"] {
        assert_eq!(preset(&value, category), (1.0, 0.95, 20, 0.0), "{category}");
    }
}

#[test]
fn flash_next_thinking_off_presets_stay_greedy() {
    let value = model_toml("qwen3.8-flash-next");
    for category in ["non_thinking", "tools"] {
        let (temperature, _, _, presence) = preset(&value, category);
        assert_eq!((temperature, presence), (0.0, 0.0), "{category}");
    }
    let behavior = value.get("behavior").expect("[behavior]");
    assert_eq!(
        behavior.get("thinking_default").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        behavior
            .get("confidence_early_stop")
            .and_then(|v| v.as_bool()),
        Some(false)
    );
}

#[test]
fn compiled_flash_next_presets_agree_with_model_toml() {
    // Meaningful in whichever target set this test binary was built for.
    for target in atlas_kernels::available_targets() {
        if target.target.model == "qwen3.8-flash-next" {
            let s = &target.sampling;
            assert_eq!(s.thinking_text.temperature, 1.0);
            assert_eq!(s.thinking_coding.temperature, 1.0);
            assert_eq!(s.non_thinking.temperature, 0.0);
            assert_eq!(s.tools.temperature, 0.0);
        }
    }
}

fn behavior_int(value: &toml::Value, key: &str) -> Option<i64> {
    value
        .get("behavior")
        .and_then(|b| b.get(key))
        .and_then(|v| v.as_integer())
}

#[test]
fn flash_next_omitted_budget_uses_unreserved_ceiling() {
    let value = model_toml("qwen3.8-flash-next");
    assert_eq!(behavior_int(&value, "max_thinking_budget"), Some(262144));
    assert_eq!(behavior_int(&value, "thinking_answer_reserve"), Some(0));
    for target in atlas_kernels::available_targets() {
        if target.target.model == "qwen3.8-flash-next" {
            assert_eq!(target.behavior.max_thinking_budget, 262144);
            assert_eq!(target.behavior.thinking_answer_reserve, 0);
            assert!(target.behavior.thinking_default);
            assert!(!target.behavior.confidence_early_stop);
        }
    }
}

#[test]
fn no_other_model_opts_into_the_answer_reserve() {
    let gb10 = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels/gb10");
    for entry in std::fs::read_dir(&gb10).unwrap() {
        let dir = entry.unwrap().path();
        let model = dir.file_name().unwrap().to_string_lossy().into_owned();
        if model == "qwen3.8-flash-next" || !dir.join("MODEL.toml").exists() {
            continue;
        }
        assert_eq!(
            behavior_int(&model_toml(&model), "thinking_answer_reserve"),
            None,
            "{model}"
        );
    }
}

#[test]
fn flash_next_thinking_tool_turns_use_the_thinking_preset() {
    let value = model_toml("qwen3.8-flash-next");
    let flag = |key: &str| {
        value
            .get("behavior")
            .and_then(|b| b.get(key))
            .and_then(|v| v.as_bool())
    };
    assert_eq!(flag("thinking_sampling_in_tools"), Some(true));
    for target in atlas_kernels::available_targets() {
        if target.target.model == "qwen3.8-flash-next" {
            assert!(target.behavior.thinking_sampling_in_tools);
        }
    }
}
