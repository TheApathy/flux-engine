// SPDX-License-Identifier: AGPL-3.0-only

//! Parse and validate tests, including the shipped recipe files.

use super::catalog::Catalog;
use super::contract;
use super::refusal::Refusal;
use super::schema::{DenseFormat, RoutePolicy, SpeculationMode};
use super::test_fixtures::*;

fn invalid(text: &str) -> String {
    match contract::parse(text, &catalog()) {
        Err(Refusal::Invalid(why)) => why,
        other => panic!("expected Invalid, got {other:?}"),
    }
}

#[test]
fn fixture_parses_with_declared_fields() {
    let l = loaded();
    assert_eq!(l.recipe.id, "test-recipe");
    assert_eq!(l.recipe.precision.lm_head, DenseFormat::Nvfp4Requant);
    assert_eq!(l.recipe.speculation.mode, SpeculationMode::Off);
    assert_eq!(l.recipe.route_policy, RoutePolicy::Flag);
    assert_eq!(
        l.declared
            .numerics
            .get("ATLAS_QWEN4_PREFILL_GDN_FI")
            .map(String::as_str),
        Some("2")
    );
    assert_eq!(l.sha256, contract::sha256_hex(RECIPE.as_bytes()));
    assert_eq!(l.sha256.len(), 64);
}

#[test]
fn unknown_keys_are_rejected() {
    let text = RECIPE.replace(
        "kv_cache_dtype = \"bf16\"",
        "kv_cache_dtype = \"bf16\"\nkv_dtyp = \"fp8\"",
    );
    assert!(invalid(&text).contains("kv_dtyp"));
}

#[test]
fn wrong_schema_version_is_rejected() {
    assert!(invalid(&RECIPE.replace("schema = 1", "schema = 2")).contains("schema 2"));
}

#[test]
fn malformed_checkpoint_sha_is_rejected() {
    assert!(invalid(&RECIPE.replace(SHA, "abc")).contains("64 hex"));
}

#[test]
fn partial_fp8_plan_is_rejected() {
    let text = RECIPE.replace("gdn_proj = \"nvfp4_requant\"", "gdn_proj = \"fp8_e4m3\"");
    assert!(invalid(&text).contains("DENSE_FP8"));
}

#[test]
fn router_has_no_fp8_path() {
    let text = RECIPE.replace("router = \"nvfp4_requant\"", "router = \"fp8_e4m3\"");
    assert!(invalid(&text).contains("router"));
}

#[test]
fn numerics_env_must_name_a_numerics_gate() {
    let text = RECIPE.replace(
        "ATLAS_QWEN4_PREFILL_GDN_FI = \"2\"",
        "ATLAS_MOE_T_LANES = \"1\"",
    );
    assert!(invalid(&text).contains("Route"));
    let text = RECIPE.replace(
        "ATLAS_QWEN4_PREFILL_GDN_FI = \"2\"",
        "ATLAS_NOT_READ = \"1\"",
    );
    assert!(invalid(&text).contains("no reader"));
}

#[test]
fn precision_selectors_cannot_hide_in_numerics_env() {
    let text = RECIPE.replace(
        "ATLAS_QWEN4_PREFILL_GDN_FI = \"2\"",
        "ATLAS_QWEN4_LMHEAD_BF16 = \"1\"",
    );
    assert!(invalid(&text).contains("[precision]"));
}

#[test]
fn fp32_state_forbids_fp16_h() {
    let text = RECIPE.replace(
        "ATLAS_QWEN4_PREFILL_GDN_FI = \"2\"",
        "ATLAS_SSM_H_FP16 = \"1\"",
    );
    assert!(invalid(&text).contains("ATLAS_SSM_H_FP16"));
}

#[test]
fn operational_rule_needs_exactly_one_form() {
    let text = RECIPE.replace("{ min = 0, max = 100 }", "{ values = [\"1\"], max = 100 }");
    assert!(invalid(&text).contains("values or min/max"));
    let text = RECIPE.replace(
        "ATLAS_SCHED_BURST_GATHER_MS = { min",
        "ATLAS_MOE_T_LANES = { min",
    );
    assert!(invalid(&text).contains("operational_env"));
}

#[test]
fn sampling_defaults_must_sit_inside_ranges() {
    let text = RECIPE.replace("temperature = 0.0", "temperature = 3.0");
    assert!(invalid(&text).contains("sampling.defaults"));
}

#[test]
fn empty_or_inverted_limits_are_rejected() {
    let text = RECIPE.replace("max_batch_size = [1, 1]", "max_batch_size = [4, 1]");
    assert!(invalid(&text).contains("max_batch_size"));
}

#[test]
fn flux_spelling_in_a_recipe_canonicalizes_to_the_atlas_reader() {
    let text = RECIPE.replace(
        "ATLAS_QWEN4_PREFILL_GDN_FI = \"2\"",
        "FLUX_QWEN4_PREFILL_GDN_FI = \"2\"",
    );
    let l = loaded_from(&text, &catalog());
    assert!(
        l.declared
            .numerics
            .contains_key("ATLAS_QWEN4_PREFILL_GDN_FI")
    );
}

#[test]
fn builtin_catalog_parses_and_classifies_known_gates() {
    let c = Catalog::builtin().expect("builtin catalog");
    use super::catalog::EnvClass;
    assert_eq!(
        c.class("ATLAS_QWEN4_PREFILL_GDN_FI"),
        Some(EnvClass::Numerics)
    );
    assert_eq!(
        c.class("ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8"),
        Some(EnvClass::Numerics)
    );
    assert_eq!(c.class("ATLAS_MOE_T_LANES"), Some(EnvClass::Route));
}

const DEFAULT: &str =
    include_str!("../../../../repro-kit/recipes/qwen38-flash-next-qad5500-hybrid-default.toml");
const HARNESS: &str =
    include_str!("../../../../repro-kit/recipes/qwen38-flash-next-qad5500-hybrid-c1-harness.toml");
const NATIVE: &str =
    include_str!("../../../../repro-kit/recipes/qwen38-flash-next-native-dense.toml");
const FP8_DENSE: &str =
    include_str!("../../../../repro-kit/recipes/qwen38-flash-next-fp8-dense.toml");

#[test]
fn shipped_recipes_parse_against_this_build() {
    let c = Catalog::builtin().expect("builtin catalog");
    let default = contract::parse(DEFAULT, &c).expect("default recipe");
    let harness = contract::parse(HARNESS, &c).expect("c1 harness recipe");
    let native = contract::parse(NATIVE, &c).expect("native-dense recipe");
    let fp8_dense = contract::parse(FP8_DENSE, &c).expect("fp8-dense recipe");
    assert_eq!(default.recipe.speculation.mode, SpeculationMode::Mtp);
    assert_eq!(default.recipe.speculation.num_drafts, Some(4));
    assert_eq!(harness.recipe.speculation.mode, SpeculationMode::Off);
    assert_eq!(harness.recipe.precision.lm_head, DenseFormat::Nvfp4Requant);
    assert_eq!(native.recipe.precision.lm_head, DenseFormat::Bf16);
    assert_eq!(fp8_dense.recipe.precision.router, DenseFormat::Bf16);
    assert_eq!(
        fp8_dense.recipe.precision.hyper_connection,
        DenseFormat::Bf16
    );
    assert_eq!(fp8_dense.recipe.precision.gdn_proj, DenseFormat::Fp8E4m3);
    assert_eq!(
        fp8_dense.recipe.precision.attention_proj,
        DenseFormat::Fp8E4m3
    );
    assert_eq!(
        fp8_dense.recipe.precision.shared_expert,
        DenseFormat::Fp8E4m3
    );
    assert_eq!(fp8_dense.recipe.precision.lm_head, DenseFormat::Bf16);
    assert_eq!(
        harness.recipe.checkpoint.sha256,
        native.recipe.checkpoint.sha256
    );
    assert_eq!(
        harness.recipe.checkpoint.sha256,
        fp8_dense.recipe.checkpoint.sha256
    );
    // Native-dense keeps the BF16 qkvz it serves from.
    let keep = |l: &contract::Loaded| {
        l.declared
            .operational
            .get("ATLAS_QWEN4_RELEASE_BF16_QKVZ")
            .cloned()
    };
    assert_eq!(
        keep(&native).and_then(|r| r.values),
        Some(vec!["0".to_string()])
    );
    assert_eq!(
        keep(&harness).and_then(|r| r.values),
        Some(vec!["1".to_string()])
    );
}

#[test]
fn shipped_native_dense_and_fp8_dense_recipes_evaluate_against_builtin_catalog() {
    use super::facts::CheckpointFacts;
    use super::precision::FamilyFormat;

    let c = Catalog::builtin().expect("builtin catalog");
    let native = contract::parse(NATIVE, &c).expect("native-dense recipe");
    let fp8_dense = contract::parse(FP8_DENSE, &c).expect("fp8-dense recipe");

    let ckpt = CheckpointFacts {
        model_type: Some("qwen4_exp".into()),
        digests: native
            .recipe
            .checkpoint
            .sha256
            .iter()
            .map(|(k, v)| (k.clone(), Ok(v.clone())))
            .collect(),
    };
    let mut s = serve();
    s.cli.insert("--kernel-target", "qwen3.8-flash-next".into());

    // 1) Native-dense recipe accepted with umbrella ATLAS_QWEN4_DENSE_NATIVE=1
    let mut native_env: std::collections::BTreeMap<String, String> = native
        .declared
        .numerics
        .iter()
        .chain(native.declared.route.iter())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    native_env.insert("ATLAS_QWEN4_DENSE_NATIVE".into(), "1".into());
    native_env.insert("ATLAS_QWEN4_RELEASE_BF16_QKVZ".into(), "0".into());
    let eff_native = contract::evaluate(&native, &native_env, &s, &ckpt, &c)
        .expect("shipped native-dense recipe accepted by builtin catalog");
    assert_eq!(
        eff_native.precision,
        vec![
            FamilyFormat {
                family: "router",
                format: "bf16".into()
            },
            FamilyFormat {
                family: "hyper_connection",
                format: "bf16".into()
            },
            FamilyFormat {
                family: "gdn_proj",
                format: "bf16".into()
            },
            FamilyFormat {
                family: "attention_proj",
                format: "bf16".into()
            },
            FamilyFormat {
                family: "shared_expert",
                format: "bf16".into()
            },
            FamilyFormat {
                family: "lm_head",
                format: "bf16".into()
            },
            FamilyFormat {
                family: "prefill_dense",
                format: "native".into()
            },
            FamilyFormat {
                family: "routed_experts",
                format: "nvfp4".into()
            },
            FamilyFormat {
                family: "gdn_state",
                format: "fp32".into()
            },
        ]
    );

    // 2) FP8-dense recipe accepted with explicit DENSE_FP8=1 + ROUTER/HC/LMHEAD_BF16=1
    let base_fp8_env: std::collections::BTreeMap<String, String> = fp8_dense
        .declared
        .numerics
        .iter()
        .chain(fp8_dense.declared.route.iter())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut explicit_fp8_env = base_fp8_env.clone();
    explicit_fp8_env.insert("ATLAS_QWEN4_DENSE_FP8".into(), "1".into());
    explicit_fp8_env.insert("ATLAS_QWEN4_ROUTER_BF16".into(), "1".into());
    explicit_fp8_env.insert("ATLAS_QWEN4_HC_BF16".into(), "1".into());
    explicit_fp8_env.insert("ATLAS_QWEN4_LMHEAD_BF16".into(), "1".into());
    let eff_fp8_explicit = contract::evaluate(&fp8_dense, &explicit_fp8_env, &s, &ckpt, &c)
        .expect("shipped fp8-dense recipe accepted with explicit BF16 + DENSE_FP8 selectors");
    let expected_fp8_precision = vec![
        FamilyFormat {
            family: "router",
            format: "bf16".into(),
        },
        FamilyFormat {
            family: "hyper_connection",
            format: "bf16".into(),
        },
        FamilyFormat {
            family: "gdn_proj",
            format: "fp8_e4m3".into(),
        },
        FamilyFormat {
            family: "attention_proj",
            format: "fp8_e4m3".into(),
        },
        FamilyFormat {
            family: "shared_expert",
            format: "fp8_e4m3".into(),
        },
        FamilyFormat {
            family: "lm_head",
            format: "bf16".into(),
        },
        FamilyFormat {
            family: "prefill_dense",
            format: "fast_fp8".into(),
        },
        FamilyFormat {
            family: "routed_experts",
            format: "nvfp4".into(),
        },
        FamilyFormat {
            family: "gdn_state",
            format: "fp32".into(),
        },
    ];
    assert_eq!(eff_fp8_explicit.precision, expected_fp8_precision);

    // 3) FP8-dense recipe accepted with umbrella DENSE_NATIVE=1 + GDN/ATTN/SHARED_BF16=0 + PREFILL_NATIVE=0 + DENSE_FP8=1
    let mut umbrella_fp8_env = base_fp8_env;
    umbrella_fp8_env.insert("ATLAS_QWEN4_DENSE_NATIVE".into(), "1".into());
    umbrella_fp8_env.insert("ATLAS_QWEN4_GDN_BF16".into(), "0".into());
    umbrella_fp8_env.insert("ATLAS_QWEN4_ATTN_BF16".into(), "0".into());
    umbrella_fp8_env.insert("ATLAS_QWEN4_SHARED_BF16".into(), "0".into());
    umbrella_fp8_env.insert("ATLAS_QWEN4_PREFILL_NATIVE".into(), "0".into());
    umbrella_fp8_env.insert("ATLAS_QWEN4_DENSE_FP8".into(), "1".into());
    let eff_fp8_umbrella = contract::evaluate(&fp8_dense, &umbrella_fp8_env, &s, &ckpt, &c)
        .expect("shipped fp8-dense recipe accepted with umbrella + 0 overrides");
    assert_eq!(eff_fp8_umbrella.precision, expected_fp8_precision);
}

#[test]
fn builtin_catalog_includes_native_dense_and_memlean_selectors() {
    use super::catalog::EnvClass;
    let c = Catalog::builtin().expect("builtin catalog");
    for name in [
        "ATLAS_QWEN4_DENSE_NATIVE",
        "ATLAS_QWEN4_DENSE_FP8",
        "ATLAS_QWEN4_ROUTER_BF16",
        "ATLAS_QWEN4_HC_BF16",
        "ATLAS_QWEN4_GDN_BF16",
        "ATLAS_QWEN4_ATTN_BF16",
        "ATLAS_QWEN4_SHARED_BF16",
        "ATLAS_QWEN4_LMHEAD_BF16",
        "ATLAS_QWEN4_PREFILL_NATIVE",
    ] {
        assert_eq!(
            c.class(name),
            Some(EnvClass::Numerics),
            "{name} must be in builtin catalog as Numerics"
        );
    }
    for name in [
        "ATLAS_QWEN4_RELEASE_MXFP8_SOURCES",
        "ATLAS_QWEN4_DENSE_DEDUP",
        "ATLAS_QWEN4_RELEASE_BF16_QKVZ",
    ] {
        assert_eq!(
            c.class(name),
            Some(EnvClass::Resource),
            "{name} must be in builtin catalog as Resource"
        );
    }
}

#[test]
fn builtin_catalog_excludes_build_scripts_and_flags_known_false_negatives() {
    use super::catalog::EnvClass;
    let c = Catalog::builtin().expect("builtin catalog");
    // Compile-time selector; at runtime only a transform-cache key, so a
    // build shell's export must not refuse a serve.
    assert_eq!(c.class("ATLAS_TARGET_QUANT"), Some(EnvClass::Resource));
    assert_eq!(c.class("ATLAS_OBFUSCATE_PTX"), None, "build.rs only");
    for name in [
        "ATLAS_TEMP_OVERRIDE",
        "ATLAS_VERIFY_SKIP_LAYERS",
        "ATLAS_REASONING_EFFORT",
        "ATLAS_ALLOW_SPECIAL_TOKENS_IN_CONTENT",
    ] {
        assert_eq!(c.class(name), Some(EnvClass::Numerics), "{name}");
    }
}

#[test]
fn builtin_catalog_has_no_prefix_fragments_dummies_or_uncataloged_builtin_recipe_env() {
    use super::catalog::EnvClass;
    let c = Catalog::builtin().expect("builtin catalog");
    for cls in [
        EnvClass::Numerics,
        EnvClass::Route,
        EnvClass::Resource,
        EnvClass::Observability,
        EnvClass::Debug,
    ] {
        for name in c.names_of(cls) {
            assert!(
                !name.ends_with('_'),
                "prefix fragment must not appear in builtin catalog: {name}"
            );
            assert!(
                name != "ATLAS_X" && name != "ATLAS_Y",
                "synthetic test dummy must not appear in builtin catalog: {name}"
            );
        }
    }
    assert_eq!(
        c.class("ATLAS_QWEN4_TARGET_LMHEAD_MODE"),
        Some(EnvClass::Numerics)
    );
    assert_eq!(
        c.class("FLUX_QWEN4_TARGET_LMHEAD_MODE"),
        Some(EnvClass::Numerics)
    );
    assert_eq!(c.class("ATLAS_TCP_BIND_PORT"), Some(EnvClass::Resource));
    for recipe in crate::recipe::builtin::recipes() {
        if let Some(profile) = crate::main_modules::model_profile::profile_for_recipe(&recipe)
            .expect("valid built-in profile")
        {
            for key in profile.env.keys() {
                assert!(
                    c.has_reader(key),
                    "built-in recipe {} sets uncataloged env var {key}",
                    recipe.id
                );
            }
        }
    }
}
