// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn one_key_contract(receipt: &str, forbidden: &str) -> String {
    format!(
        "atlas:\n  model_type: test\n  environment_policy: sealed-v1\n  profile_kind: test-profile\n  environment_sha256: {receipt}\n  forbidden_env:\n    - {forbidden}\n  env:\n    ATLAS_X: \"1\"\n"
    )
}

#[test]
fn malformed_sealed_profiles_are_parse_errors() {
    let good = "970c73cd2114d1d9bf66fc2b8fce4580f66c8f4bb7d1904a1dc1c0e980af06f3";
    let mismatch = one_key_contract(&"0".repeat(64), "ATLAS_Y");
    let error = parse_profile("test/mismatch", &mismatch)
        .expect_err("a stale receipt must fail parsing")
        .to_string();
    assert!(error.contains("receipt mismatch"), "{error}");

    let overlap = one_key_contract(good, "ATLAS_X");
    let error = parse_profile("test/overlap", &overlap)
        .expect_err("a required key cannot also be forbidden")
        .to_string();
    assert!(error.contains("both required and forbidden"), "{error}");

    let partial = "atlas:\n  model_type: test\n  profile_kind: test-profile\n";
    let error = parse_profile("test/partial", partial)
        .expect_err("a partial contract must fail closed")
        .to_string();
    assert!(error.contains("environment_policy is required"), "{error}");

    let bad_name = one_key_contract(good, "ATLAS_Y").replace("ATLAS_X:", "atlas_x:");
    let error = parse_profile("test/bad-name", &bad_name)
        .expect_err("sealed keys must be canonical ATLAS names")
        .to_string();
    assert!(error.contains("invalid sealed environment name"), "{error}");
}

#[test]
fn flash_next_profiles_seal_distinct_memory_contracts() {
    let recipes = crate::recipe::builtin::recipes();
    let get = |id: &str| {
        let recipe = recipes
            .iter()
            .find(|recipe| recipe.id == id)
            .unwrap_or_else(|| panic!("missing {id}"));
        profile_for_recipe(recipe)
            .unwrap_or_else(|error| panic!("invalid profile {id}: {error:#}"))
            .unwrap_or_else(|| panic!("missing profile {id}"))
    };
    let prefill = get("qwen3.8/qwen3.8-flash-next-prefill4k-champion");
    let mtp = get("qwen3.8/qwen3.8-flash-next-native-mtp-safe-local");
    let cache = "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE";
    let reserve = "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB";
    assert_eq!(prefill.env.len(), 58, "measured environment census");
    assert_eq!(prefill.env.get(cache).map(String::as_str), Some("1"));
    assert_eq!(mtp.env.get(cache).map(String::as_str), Some("0"));
    for profile in [&prefill, &mtp] {
        let receipt = profile
            .validate_environment_with(|name| Ok(profile.env.get(name).cloned()), None)
            .expect("exact environment")
            .expect("sealed receipt");
        assert_eq!(receipt.profile_kind, profile.profile_kind().unwrap());
        let error = profile
            .validate_environment_with(
                |name| {
                    Ok((name == reserve)
                        .then(|| "4".to_string())
                        .or_else(|| profile.env.get(name).cloned()))
                },
                None,
            )
            .expect_err("the unmeasured reserve must be refused")
            .to_string();
        assert!(error.contains(reserve), "{error}");
    }
    let error = prefill
        .validate_environment_with(
            |name| {
                Ok((name == cache)
                    .then(|| "0".to_string())
                    .or_else(|| prefill.env.get(name).cloned()))
            },
            None,
        )
        .expect_err("cache-off cannot masquerade as prefill")
        .to_string();
    assert!(error.contains(cache), "{error}");
}

#[test]
fn dense_dflash_generations_are_separate_build_tracks() {
    let recipes = crate::recipe::builtin::recipes();
    let mut profiles = Vec::new();
    for generation in ["v14", "v15"] {
        let id = format!("qwen3.8/qwen3.8-27b-dflash2-{generation}-safe-local");
        let recipe = recipes
            .iter()
            .find(|recipe| recipe.id == id)
            .unwrap_or_else(|| panic!("missing {id}"));
        let profile = profile_for_recipe(recipe)
            .unwrap_or_else(|error| panic!("invalid profile {id}: {error:#}"))
            .unwrap_or_else(|| panic!("missing {id}"));
        let track = format!("dense-qwen-dflash-{generation}");
        assert_eq!(profile.profile_kind(), Some(track.as_str()));
        let receipt = profile
            .validate_environment_with(
                |name| Ok(profile.env.get(name).cloned()),
                Some(track.as_str()),
            )
            .expect("matching track")
            .expect("sealed receipt");
        assert_eq!(receipt.build_track.as_deref(), Some(track.as_str()));
        let wrong = format!(
            "dense-qwen-dflash-{}",
            if generation == "v14" { "v15" } else { "v14" }
        );
        let error = profile
            .validate_environment_with(|name| Ok(profile.env.get(name).cloned()), Some(&wrong))
            .expect_err("cross-generation binary must be refused")
            .to_string();
        assert!(error.contains("separately sealed binary"), "{error}");
        profiles.push(profile);
    }
    assert_eq!(
        profiles[0].env, profiles[1].env,
        "only source generation differs"
    );
}

fn v15_dense_dflash_profile() -> ModelProfile {
    let recipes = crate::recipe::builtin::recipes();
    let recipe = recipes
        .iter()
        .find(|recipe| recipe.id == "qwen3.8/qwen3.8-27b-dflash2-v15-safe-local")
        .expect("v15 dense DFlash recipe");
    profile_for_recipe(recipe)
        .expect("valid profile")
        .expect("sealed profile")
}

#[test]
fn dense_dflash_seal_admits_only_the_reviewed_lookup_overlay() {
    let profile = v15_dense_dflash_profile();
    let track = Some("dense-qwen-dflash-v15");
    assert_eq!(
        profile.env.get("ATLAS_DFLASH_SAM").map(String::as_str),
        Some("0")
    );
    let with = |lookup: Option<&'static str>, sam: &'static str| {
        let profile = &profile;
        move |name: &str| -> anyhow::Result<Option<String>> {
            Ok(match name {
                "ATLAS_DFLASH_LOOKUP" => lookup.map(str::to_string),
                "ATLAS_DFLASH_SAM" => Some(sam.to_string()),
                _ => profile.env.get(name).cloned(),
            })
        }
    };

    let base = profile
        .validate_environment_with(with(None, "0"), track)
        .expect("sealed base")
        .expect("receipt");
    assert_eq!(base.lookup, None);

    let sam = profile
        .validate_environment_with(with(Some("sam"), "1"), track)
        .expect("reviewed SAM overlay")
        .expect("receipt");
    assert_eq!(sam.lookup, Some("sam"));
    assert_eq!(
        sam.environment_sha256, base.environment_sha256,
        "the receipt keeps hashing the declared base"
    );

    for (lookup, value, culprit) in [
        (None, "1", "ATLAS_DFLASH_SAM"),
        (Some("off"), "1", "ATLAS_DFLASH_SAM"),
        (Some("sam"), "0", "ATLAS_DFLASH_SAM"),
        (Some("retrieval"), "1", "ATLAS_DFLASH_SAM"),
        (Some("on"), "1", "ATLAS_DFLASH_LOOKUP"),
    ] {
        let error = profile
            .validate_environment_with(with(lookup, value), track)
            .expect_err("unreviewed SAM state must break the seal")
            .to_string();
        assert!(error.contains(culprit), "{lookup:?} SAM={value}: {error}");
    }
}

#[test]
fn profiles_without_lookup_controls_ignore_the_selector() {
    let recipes = crate::recipe::builtin::recipes();
    let recipe = recipes
        .iter()
        .find(|recipe| recipe.id == "qwen3.8/qwen3.8-flash-next-native-mtp-safe-local")
        .expect("Flash-Next MTP recipe");
    let profile = profile_for_recipe(recipe)
        .expect("valid profile")
        .expect("sealed profile");
    assert!(!profile.env.contains_key("ATLAS_DFLASH_SAM"));
    let receipt = profile
        .validate_environment_with(
            |name| {
                Ok((name == "ATLAS_DFLASH_LOOKUP")
                    .then(|| "malformed".to_string())
                    .or_else(|| profile.env.get(name).cloned()))
            },
            None,
        )
        .expect("Flash-Next never reads the DFlash lookup selector")
        .expect("receipt");
    assert_eq!(receipt.lookup, None);
}
