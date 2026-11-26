// SPDX-License-Identifier: AGPL-3.0-only

//! The build-time `[[dflash.profile]]` parser fails the build on any row it
//! would otherwise half-check.

#[path = "../build_profile.rs"]
mod build_profile;

use std::path::{Path, PathBuf};

use build_profile::generate_drafter_profiles_rs;

fn hw_dir(name: &str, models: &[(&str, &str)]) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("atlas-build-profile-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    for (model, toml) in models {
        std::fs::create_dir_all(dir.join(model)).unwrap();
        std::fs::write(dir.join(model).join("MODEL.toml"), toml).unwrap();
    }
    dir
}

fn panic_message(dir: &Path) -> String {
    let owned = dir.to_owned();
    let payload = std::panic::catch_unwind(move || generate_drafter_profiles_rs(&owned))
        .expect_err("malformed row was accepted");
    let _ = std::fs::remove_dir_all(dir);
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default()
}

const EXACT: &str = r#"
[[dflash.profile]]
name = "row"
kind = "dflash2"
admission = "exact"
verify_k = [2, 8]
architecture = "DFlash2DraftModel"
hidden_size = 5120
intermediate_size = 17408
num_hidden_layers = 5
num_attention_heads = 32
num_key_value_heads = 8
head_dim = 128
vocab_size = 248320
block_size = 8
mask_token_id = 248070
target_layer_ids = [5, 19, 33, 47, 61]
conv_kernel_size = 2
conv_group_size = 16
selector_rank = 256
selector_top_k = 16
is_causal = false
"#;

#[test]
fn exact_row_renders_every_pinned_field() {
    let dir = hw_dir(
        "exact",
        &[("m", EXACT), ("no-profile", "[model]\nname = \"x\"\n")],
    );
    let generated = generate_drafter_profiles_rs(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(generated.matches("DrafterProfile {").count(), 1);
    for needle in [
        "target_model: \"m\"",
        "kind: DrafterKind::Dflash2",
        "admission: DrafterAdmission::Exact",
        "verify_k: Some((2, 8))",
        "target_layer_ids: Some(&[5, 19, 33, 47, 61])",
        "mask_token_id: Some(248070)",
        "draft_vocab_size: VocabPin::Unchecked",
        "capture_mode: DrafterCaptureMode::ResidualSlice",
    ] {
        assert!(generated.contains(needle), "{needle}\n{generated}");
    }
}

#[test]
fn vocab_pins_distinguish_undeclared_from_equals() {
    let toml = format!("{EXACT}draft_vocab_size = 0\nselector_vocab_size = 248077\n");
    let dir = hw_dir("pins", &[("m", &toml)]);
    let generated = generate_drafter_profiles_rs(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(generated.contains("draft_vocab_size: VocabPin::Undeclared"));
    assert!(generated.contains("selector_vocab_size: VocabPin::Equals(248077)"));
}

#[test]
fn malformed_rows_fail_the_build() {
    let generic = "[[dflash.profile]]\nname = \"g\"\nkind = \"dflash\"\nadmission = \"generic\"\n";
    let cases: Vec<(&str, String, &str)> = vec![
        (
            "unknown key",
            format!("{EXACT}taget_layer_ids = [1]\n"),
            "unknown key",
        ),
        (
            "generic pins",
            format!("{generic}hidden_size = 5120\n"),
            "must not pin",
        ),
        (
            "missing pin",
            EXACT.replace("head_dim = 128\n", ""),
            "must pin `head_dim`",
        ),
        (
            "verify_k past block",
            EXACT.replace("verify_k = [2, 8]", "verify_k = [2, 16]"),
            "verify_k",
        ),
        (
            "unqualified default",
            format!("{generic}launcher_default = true\n"),
            "qualified",
        ),
        (
            "two defaults",
            format!(
                "{generic}qualified = true\nlauncher_default = true\n{}qualified = true\nlauncher_default = true\n",
                generic.replace("\"g\"", "\"h\"")
            ),
            "at most one",
        ),
        ("duplicate name", format!("{generic}{generic}"), "duplicate"),
        (
            "quorum without depth",
            EXACT.replace("\"exact\"", "\"quorum\""),
            "num_target_layers",
        ),
        (
            "quorum causal",
            EXACT
                .replace("\"exact\"", "\"quorum\"")
                .replace("is_causal = false", "is_causal = true")
                + "num_target_layers = 64\n",
            "is_causal = false",
        ),
        (
            "unknown capture mode",
            format!("{generic}capture_mode = \"fp32\"\n"),
            "capture_mode",
        ),
        (
            "modelopt off-census",
            EXACT.replace("conv_group_size = 16", "conv_group_size = 8")
                + "allow_modelopt_nvfp4 = true\n",
            "allow_modelopt_nvfp4",
        ),
        (
            "hyper projection without target identity",
            format!("{EXACT}capture_mode = \"qwen4_hyper_projected\"\n"),
            "target identity",
        ),
    ];
    for (name, toml, expected) in cases {
        let message = panic_message(&hw_dir(name.replace(' ', "-").as_str(), &[("m", &toml)]));
        assert!(message.contains(expected), "{name}: {message}");
    }
}

#[test]
fn the_shipped_kernel_tree_parses() {
    let gb10 = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/gb10");
    let generated = generate_drafter_profiles_rs(&gb10);
    for needle in [
        "target_model: \"qwen3.8-27b\",\n        name: \"zlab-dflash2\"",
        "target_model: \"qwen3.8-flash-next\",\n        name: \"native-v3\"",
        "target_model: \"qwen3.8-flash-next\",\n        name: \"native-dflash2\"",
        "target_model: \"glm5.3-flash\",\n        name: \"glm-dflash2\"",
    ] {
        assert!(generated.contains(needle), "{needle}");
    }
    assert!(!generated.contains("target_model: \"gemma-4-31b\""));
    assert!(!generated.contains("target_model: \"deepseek-v4.1\""));
}
