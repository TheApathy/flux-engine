// SPDX-License-Identifier: AGPL-3.0-only

//! Recipes shipped INSIDE the binary, for models the public index does not carry yet.
//!
//! The Library lists a checkpoint only as well as it knows a recipe for it; a model with no
//! recipe gets a "starting point" guessed from a donor, and for DeepSeek-V4.1 any donor guess
//! is wrong (its memory needs, its expert-pack env and its run-alone admission exist nowhere
//! else). So the measured configuration ships here, in the same YAML format as the index, and
//! [`with_builtins`] adds it beneath the fetched recipes — a published recipe with the same id
//! wins, so this steps aside the moment upstream carries one.
//!
//! The YAML may carry an `atlas:` block (model_type, memory, run-alone, env) that
//! [`Recipe::parse`] ignores and `main_modules::model_profile` reads.

use super::Recipe;

/// `(id, yaml)` for every built-in recipe. The id is `family/stem`, like the index's.
pub const BUILTIN_YAML: &[(&str, &str)] = &[
    (
        "deepseek/deepseek-v4.1-flash-next",
        include_str!("builtin/deepseek-v4.1-flash-next.yaml"),
    ),
    // The all-models matrix configurations, with each model pinned to its tuned kernel
    // target and carrying the environment its measured numbers need.
    (
        "qwen3.8/qwen3.8-27b-optimized-local",
        include_str!("builtin/qwen3.8-27b-optimized-local.yaml"),
    ),
    (
        "qwen3.8/qwen3.8-27b-dflash2-v14-safe-local",
        include_str!("builtin/qwen3.8-27b-dflash2-safe-local.yaml"),
    ),
    (
        "qwen3.8/qwen3.8-27b-dflash2-v15-safe-local",
        include_str!("builtin/qwen3.8-27b-dflash2-v15-safe-local.yaml"),
    ),
    (
        "qwen3.8/qwen3.8-flash-next-offload-local",
        include_str!("builtin/qwen3.8-flash-next-offload-local.yaml"),
    ),
    (
        "qwen3.8/qwen3.8-flash-next-prefill4k-champion",
        include_str!("builtin/qwen3.8-flash-next-prefill4k-champion.yaml"),
    ),
    (
        "qwen3.8/qwen3.8-flash-next-native-mtp-safe-local",
        include_str!("builtin/qwen3.8-flash-next-native-mtp-safe-local.yaml"),
    ),
    (
        "glm5.3/glm-5.3-flash-exl3-local",
        include_str!("builtin/glm-5.3-flash-exl3-local.yaml"),
    ),
    (
        "glm5.3/glm-5.3-flash-exl3-dflash2-local",
        include_str!("builtin/glm-5.3-flash-exl3-dflash2-local.yaml"),
    ),
    (
        "qwen3.6/aeon-q36-27b-full-local",
        include_str!("builtin/aeon-q36-27b-full-local.yaml"),
    ),
];

/// Every built-in recipe, parsed. A built-in that fails to parse is a build defect and is
/// caught by the tests below, so it is skipped with an error log rather than taking the
/// Library down.
pub fn recipes() -> Vec<Recipe> {
    BUILTIN_YAML
        .iter()
        .filter_map(|(id, text)| match Recipe::parse(*id, text) {
            Ok(mut r) => {
                r.built_in = true;
                Some(r)
            }
            Err(e) => {
                tracing::error!("built-in recipe {id} does not parse: {e:#}");
                None
            }
        })
        .collect()
}

/// `fetched` plus every built-in whose id it does not already carry.
pub fn with_builtins(fetched: &[Recipe]) -> Vec<Recipe> {
    with_builtins_for(fetched, |_| true)
}

/// [`with_builtins`], restricted to built-ins whose `model` passes `present` — the Library
/// adds one only when its checkpoint is on disk.
pub fn with_builtins_for(fetched: &[Recipe], present: impl Fn(&str) -> bool) -> Vec<Recipe> {
    let mut all = fetched.to_vec();
    for b in recipes() {
        if present(&b.model) && !all.iter().any(|r| r.id == b.id) {
            all.push(b);
        }
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_parses_and_is_atlas() {
        let all = recipes();
        assert_eq!(
            all.len(),
            BUILTIN_YAML.len(),
            "a built-in recipe failed to parse"
        );
        for r in &all {
            assert!(r.is_atlas(), "{} must be runtime: atlas", r.id);
            assert_eq!(
                r.defaults.get("host").map(String::as_str),
                Some("127.0.0.1"),
                "{} must default to localhost bind",
                r.id
            );
        }
    }

    #[test]
    fn deepseek_v41_builds_a_valid_serve_command() {
        let r = recipes()
            .into_iter()
            .find(|r| r.id == "deepseek/deepseek-v4.1-flash-next")
            .unwrap();
        assert_eq!(r.model, "DeepSeek-V4.1-Flash-Next-DGX-Spark-512K");
        let args = r
            .serve_args(&Default::default())
            .expect("argv must pass clap and validation");
        assert_eq!(args.max_seq_len, 8192);
        assert_eq!(args.max_batch_size, 1);
    }

    /// Every built-in renders a command line clap accepts, and the two qwen3_5/5120
    /// checkpoints (which the auto-pick would both put on qwen3.6-27b) are pinned to their
    /// own targets.
    #[test]
    fn every_builtin_builds_a_valid_serve_command_with_its_kernel_target() {
        let want = [
            ("optimized-qwen", Some("qwen3.8-27b")),
            (
                "Qwen3.8-Flash-Next-NVFP4-Offload",
                Some("qwen3.8-flash-next"),
            ),
            ("AEON-Q36-27B-Full", Some("qwen3.6-27b")),
            ("GLM-5.3-Flash-exl3-2.05bpw", None),
        ];
        let all = recipes();
        for (model, target) in want {
            let r = all
                .iter()
                .find(|r| r.model == model)
                .unwrap_or_else(|| panic!("{model}"));
            let args = r
                .serve_args(&Default::default())
                .unwrap_or_else(|e| panic!("{model}: {e:#}"));
            assert_eq!(args.kernel_target.as_deref(), target, "{model}");
        }
    }

    #[test]
    fn safe_qwen38_profiles_pin_checkpoint_geometry_and_bounded_copies() {
        let all = recipes();
        let dflash = all
            .iter()
            .find(|r| r.id == "qwen3.8/qwen3.8-27b-dflash2-v14-safe-local")
            .expect("official DFlash2 profile");
        let dflash_args = dflash
            .serve_args(&Default::default())
            .expect("valid DFlash2 argv");
        assert_eq!(dflash.model, "q27-nvidia-nvfp4");
        assert!(dflash_args.dflash);
        assert_eq!(
            dflash_args.draft_model.as_deref(),
            Some("/models/drafter-dflash2-incoai")
        );
        assert_eq!(dflash_args.dflash_gamma, Some(7));
        assert_eq!(dflash_args.dflash_window_size, 4096);
        assert_eq!(dflash_args.dflash_quantization, "nvfp4");
        assert_eq!(dflash_args.dflash_draft_sample_method, "greedy");
        assert_eq!(dflash_args.dflash_method, "flat");
        let dflash_profile = crate::main_modules::model_profile::profile_for_recipe(dflash)
            .expect("valid built-in profile")
            .expect("exact DFlash2 launch profile");
        crate::cli::validate_dflash_runtime_environment_with(&dflash_args, |name| {
            dflash_profile.env.get(name).cloned()
        })
        .expect("the exact built-in DFlash2 environment is release-flat");
        let inherited_error =
            crate::cli::validate_dflash_runtime_environment_with(&dflash_args, |name| {
                (name == "ATLAS_DFLASH_BRANCH")
                    .then(|| "1".to_string())
                    .or_else(|| dflash_profile.env.get(name).cloned())
            })
            .expect_err(
                "an inherited branch override must defeat neither recipe nor TUI admission",
            );
        assert!(inherited_error.contains("ATLAS_DFLASH_BRANCH"));
        assert_eq!(
            dflash_profile
                .env
                .get("ATLAS_DFLASH_METHOD")
                .map(String::as_str),
            Some("flat")
        );
        assert!(
            dflash_profile
                .env
                .keys()
                .all(|key| !key.starts_with("ATLAS_DDTREE_")),
            "the flat release profile must not carry an apparent DDTree control surface"
        );

        let mtp = all
            .iter()
            .find(|r| r.id == "qwen3.8/qwen3.8-flash-next-native-mtp-safe-local")
            .expect("safe native-MTP profile");
        let mtp_args = mtp
            .serve_args(&Default::default())
            .expect("valid native-MTP argv");
        assert!(mtp_args.speculative);
        assert_eq!(
            mtp_args.mtp_from_path.as_deref(),
            Some(std::path::Path::new(
                "/models/Qwen3.8-Flash-Next-Inferact-MTP"
            ))
        );
        assert_eq!(mtp_args.num_drafts, 4);
        assert_eq!(
            mtp_args.mtp_vocab, 0,
            "native MTP qualification must use full vocab"
        );
        let mtp_profile = crate::main_modules::model_profile::profile_for_recipe(mtp)
            .expect("valid built-in profile")
            .expect("exact native-MTP launch profile");
        let cache = "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE";
        let reserve = "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB";
        let ple_timeout = "ATLAS_QWEN4_PLE_IO_TIMEOUT_MS";
        assert_eq!(mtp_profile.env.get(cache).map(String::as_str), Some("0"));
        assert!(
            !mtp_profile.env.contains_key(reserve),
            "a disabled resident cache must not advertise a reserve-based safety contract"
        );
        assert_eq!(
            mtp_profile.env.get(ple_timeout).map(String::as_str),
            Some("30000")
        );
        assert!(
            !dflash_profile.env.contains_key(reserve),
            "the MoE cache reserve must not leak into dense DFlash2"
        );
        let reserve_count = all
            .iter()
            .filter(|recipe| {
                crate::main_modules::model_profile::profile_for_recipe(recipe)
                    .expect("valid built-in profile")
                    .is_some_and(|profile| profile.env.contains_key(reserve))
            })
            .count();
        // Under environment_policy sealed-v1 every built-in lists the reserve in
        // `forbidden_env` (model_profile_sealed_tests refuses it for both
        // Flash-Next profiles); none advertises it in `env`.
        assert_eq!(reserve_count, 0, "no built-in advertises the reserve");

        for recipe in [dflash, mtp] {
            let profile = crate::main_modules::model_profile::profile_for_recipe(recipe)
                .expect("valid built-in profile")
                .expect("exact built-in launch profile");
            assert_eq!(
                profile
                    .env
                    .get("ATLAS_BOUNDED_HOST_COPY")
                    .map(String::as_str),
                Some("1"),
                "{}",
                recipe.id
            );
            assert_eq!(
                profile
                    .env
                    .get("ATLAS_H2D_LEGACY_DRAIN")
                    .map(String::as_str),
                Some("0"),
                "{}",
                recipe.id
            );
            assert_eq!(
                profile.env.get("ATLAS_DFLASH_ASYNC").map(String::as_str),
                Some("0"),
                "{}",
                recipe.id
            );
        }
    }

    #[test]
    fn release_dflash_profiles_build_only_the_nvfp4_lm_head() {
        let all = recipes();
        for id in [
            "qwen3.8/qwen3.8-27b-optimized-local",
            "qwen3.8/qwen3.8-27b-dflash2-v14-safe-local",
            "qwen3.8/qwen3.8-27b-dflash2-v15-safe-local",
        ] {
            let recipe = all
                .iter()
                .find(|recipe| recipe.id == id)
                .unwrap_or_else(|| panic!("missing built-in {id}"));
            let args = recipe
                .serve_args(&Default::default())
                .unwrap_or_else(|error| panic!("{id}: {error:#}"));
            let profile = crate::main_modules::model_profile::profile_for_recipe(recipe)
                .unwrap_or_else(|error| panic!("invalid profile for {id}: {error:#}"))
                .unwrap_or_else(|| panic!("missing profile for {id}"));
            assert_eq!(
                profile
                    .env
                    .get("ATLAS_DFLASH_LM_HEAD_NVFP4")
                    .map(String::as_str),
                Some("1"),
                "{id}"
            );
            assert_eq!(
                profile
                    .env
                    .get("ATLAS_DFLASH_LM_HEAD_FP8")
                    .map(String::as_str),
                Some("0"),
                "{id}"
            );
            crate::cli::validate_dflash_runtime_environment_with(&args, |name| {
                profile.env.get(name).cloned()
            })
            .unwrap_or_else(|error| panic!("{id}: {error}"));
        }
    }

    /// A published recipe with the same id wins; a different id is additive.
    #[test]
    fn a_fetched_recipe_with_the_same_id_replaces_the_builtin() {
        let mut fetched = recipes();
        fetched[0].description = "published upstream".into();
        let merged = with_builtins(&fetched);
        assert_eq!(merged.len(), fetched.len());
        assert_eq!(merged[0].description, "published upstream");
        // Control: with nothing fetched, the built-ins appear.
        assert_eq!(with_builtins(&[]).len(), BUILTIN_YAML.len());
        // And only for checkpoints that are present.
        assert!(with_builtins_for(&[], |_| false).is_empty());
    }
}
