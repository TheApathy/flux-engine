// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for [`crate::flux_config`].

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use super::*;

fn env_map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

#[test]
fn empty_map_matches_default() {
    let cfg = FluxConfig::from_map(&BTreeMap::new()).expect("empty map parses");
    assert_eq!(cfg, FluxConfig::default());
    assert_eq!(cfg.mode, "auto");
    assert_eq!(cfg.host, "127.0.0.1");
    assert_eq!(cfg.port, 8899);
    assert_eq!(cfg.concurrency, 1);
    assert!((cfg.gpu_memory_utilization - 0.55).abs() < f32::EPSILON);
    assert_eq!(cfg.mem_floor_mib, 6_656);
    assert!(!cfg.disable_speculation);
    assert!(!cfg.ple_mmap);
    assert!(!cfg.debug);
    assert!(!cfg.profile);
}

#[test]
fn parses_all_tier1_operator_fields_from_flux_keys() {
    let map = env_map(&[
        ("FLUX_TARGET_HW", "gb10"),
        ("FLUX_TARGET_MODEL", "qwen3.8-flash-next"),
        ("FLUX_TARGET_QUANT", "nvfp4"),
        ("FLUX_MODEL", "/models/Qwen3.8-Flash-Next-NVFP4-Offload"),
        ("FLUX_MODE", "interactive"),
        ("FLUX_RECIPE", "/etc/flux/recipe.toml"),
        ("FLUX_WEIGHT_CACHE_DIR", "/var/cache/flux-weight-cache"),
        ("FLUX_MAX_SEQ_LEN", "131072"),
        ("FLUX_CONCURRENCY", "8"),
        ("FLUX_MAX_BATCH_TOKENS", "16384"),
        ("FLUX_QWEN4_PREFILL_WIDE", "4096"),
        ("FLUX_QWEN4_NATIVE_BATCH", "1"),
        ("FLUX_QUEUE_TIMEOUT", "45"),
        ("FLUX_KV_CACHE_DTYPE", "bf16"),
        ("FLUX_KV_POOL_TOKENS", "65536"),
        ("FLUX_KV_MAX_TOKENS", "131072"),
        ("FLUX_KV_POOL_UNCAPPED", "1"),
        ("FLUX_GPU_MEMORY_UTILIZATION", "0.52"),
        ("FLUX_MEM_WATCH", "on"),
        ("FLUX_MEM_FLOOR_MIB", "8192"),
        ("FLUX_MEMAVAILABLE_FLOOR_MIB", "7168"),
        ("FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB", "10"),
        ("FLUX_DISABLE_SPECULATION", "1"),
        ("FLUX_DRAFT_MODEL", "/models/Inferact-MTP"),
        ("FLUX_GAMMA", "4"),
        ("FLUX_MTP_SOURCE", "sidecar"),
        ("FLUX_DFLASH_DRAFT_CAP", "15"),
        ("FLUX_DFLASH_LOOKUP", "sam"),
        ("FLUX_PLE_MMAP", "true"),
        ("FLUX_PLE_CACHE_MB", "4096"),
        ("FLUX_PLE_PARKED_CACHE_MB", "2048"),
        ("FLUX_HOST", "0.0.0.0"),
        ("FLUX_PORT", "9000"),
        ("FLUX_API_KEY", "sk-secret-token"),
        ("FLUX_TRUST_PROXY", "1"),
        ("FLUX_SETTINGS_FILE", "/var/lib/flux/settings.json"),
        ("FLUX_HOME", "/var/lib/flux"),
        ("FLUX_DEBUG", "on"),
        ("FLUX_LOAD_DEBUG", "always"),
        ("FLUX_PROFILE", "1"),
    ]);

    let cfg = FluxConfig::from_map(&map).expect("full config parses");
    assert_eq!(cfg.target_hw.as_deref(), Some("gb10"));
    assert_eq!(cfg.target_model.as_deref(), Some("qwen3.8-flash-next"));
    assert_eq!(cfg.target_quant.as_deref(), Some("nvfp4"));
    assert_eq!(
        cfg.model,
        Some(PathBuf::from("/models/Qwen3.8-Flash-Next-NVFP4-Offload"))
    );
    assert_eq!(cfg.mode, "interactive");
    assert_eq!(cfg.recipe, Some(PathBuf::from("/etc/flux/recipe.toml")));
    assert_eq!(
        cfg.weight_cache_dir,
        Some(PathBuf::from("/var/cache/flux-weight-cache"))
    );
    assert_eq!(cfg.max_seq_len, Some(131_072));
    assert_eq!(cfg.concurrency, 8);
    assert_eq!(cfg.max_batch_tokens, Some(16_384));
    assert_eq!(cfg.qwen4_prefill_wide, Some(4_096));
    assert_eq!(cfg.qwen4_native_batch, NativeBatchMode::ForceNative);
    assert_eq!(cfg.queue_timeout_secs, 45);
    assert_eq!(cfg.kv_cache_dtype, KvCacheDtype::Bf16);
    assert_eq!(cfg.kv_pool_tokens, Some(65_536));
    assert_eq!(cfg.kv_max_tokens, Some(131_072));
    assert!(cfg.kv_pool_uncapped);
    assert!((cfg.gpu_memory_utilization - 0.52).abs() < 1e-6);
    assert_eq!(cfg.mem_watch, MemWatchMode::On);
    assert_eq!(cfg.mem_floor_mib, 8_192);
    assert_eq!(cfg.memavailable_floor_mib, Some(7_168));
    assert_eq!(cfg.fp8_weight_cache_reserve_gb, Some(10));
    assert!(cfg.disable_speculation);
    assert_eq!(cfg.draft_model, Some(PathBuf::from("/models/Inferact-MTP")));
    assert_eq!(cfg.gamma, Some(4));
    assert_eq!(cfg.mtp_source, MtpSource::Sidecar);
    assert_eq!(cfg.dflash_draft_cap, Some(15));
    assert_eq!(cfg.dflash_lookup, DflashLookupMode::Sam);
    assert!(cfg.ple_mmap);
    assert_eq!(cfg.ple_cache_mb, Some(4_096));
    assert_eq!(cfg.ple_parked_cache_mb, Some(2_048));
    assert_eq!(cfg.host, "0.0.0.0");
    assert_eq!(cfg.port, 9000);
    assert_eq!(
        cfg.api_key.as_ref().map(SecretString::expose),
        Some("sk-secret-token")
    );
    assert!(cfg.trust_proxy);
    assert_eq!(
        cfg.settings_file,
        SettingsFileMode::Custom(PathBuf::from("/var/lib/flux/settings.json"))
    );
    assert_eq!(cfg.home, Some(PathBuf::from("/var/lib/flux")));
    assert!(cfg.debug);
    assert_eq!(cfg.load_debug, LoadDebugMode::Always);
    assert!(cfg.profile);
}

#[test]
fn atlas_alias_beats_flux_spelling_and_shorthands_resolve() {
    let map = env_map(&[
        ("FLUX_MAX_BATCH_TOKENS", "4096"),
        ("ATLAS_MAX_BATCH_TOKENS", "8192"),
        ("FLUX_PREFILL_WIDE", "4096"),
        ("FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB", "9"),
        (
            "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB",
            "12",
        ),
        ("FLUX_MTP_SOURCE", "checkpoint"),
        ("ATLAS_MTP_SOURCE", "sidecar"),
    ]);
    let cfg = FluxConfig::from_map(&map).expect("alias map parses");
    assert_eq!(cfg.max_batch_tokens, Some(8_192));
    assert_eq!(cfg.qwen4_prefill_wide, Some(4_096));
    assert_eq!(cfg.fp8_weight_cache_reserve_gb, Some(12));
    assert_eq!(cfg.mtp_source, MtpSource::Sidecar);
    assert!(is_public_operator_var("FLUX_PREFILL_WIDE"));
    assert!(is_public_operator_var(
        "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB"
    ));
}

#[test]
fn api_key_is_redacted_in_debug_output() {
    let map = env_map(&[("FLUX_API_KEY", "super-secret-value")]);
    let cfg = FluxConfig::from_map(&map).expect("parses");
    let rendered = format!("{cfg:?}");
    assert!(
        !rendered.contains("super-secret-value"),
        "Debug leaked api_key: {rendered}"
    );
    assert!(rendered.contains("<redacted>"), "{rendered}");
}

#[test]
fn rejects_invalid_and_out_of_range_values() {
    assert!(matches!(
        FluxConfig::from_map(&env_map(&[("FLUX_PORT", "0")])),
        Err(FluxConfigError::OutOfRange {
            key: "FLUX_PORT",
            ..
        })
    ));
    assert!(matches!(
        FluxConfig::from_map(&env_map(&[("FLUX_GPU_MEMORY_UTILIZATION", "1.5")])),
        Err(FluxConfigError::OutOfRange {
            key: "FLUX_GPU_MEMORY_UTILIZATION",
            ..
        })
    ));
    assert!(matches!(
        FluxConfig::from_map(&env_map(&[("FLUX_GPU_MEMORY_UTILIZATION", "NaN")])),
        Err(FluxConfigError::OutOfRange {
            key: "FLUX_GPU_MEMORY_UTILIZATION",
            ..
        })
    ));
    assert!(matches!(
        FluxConfig::from_map(&env_map(&[("FLUX_KV_CACHE_DTYPE", "int2")])),
        Err(FluxConfigError::InvalidValue {
            key: "FLUX_KV_CACHE_DTYPE",
            ..
        })
    ));
    assert!(matches!(
        FluxConfig::from_map(&env_map(&[("FLUX_CONCURRENCY", "0")])),
        Err(FluxConfigError::OutOfRange {
            key: "FLUX_CONCURRENCY",
            ..
        })
    ));
    assert!(matches!(
        FluxConfig::from_map(&env_map(&[
            ("FLUX_KV_POOL_TOKENS", "131072"),
            ("FLUX_KV_MAX_TOKENS", "65536"),
        ])),
        Err(FluxConfigError::Conflict { .. })
    ));
    assert!(matches!(
        FluxConfig::from_map(&env_map(&[
            ("FLUX_QWEN4_NATIVE_BATCH", "1"),
            ("FLUX_CONCURRENCY", "64"),
        ])),
        Err(FluxConfigError::Conflict { .. })
    ));
}

#[test]
fn public_operator_spec_table_is_well_formed_and_unique() {
    let mut names = BTreeSet::new();
    let mut aliases = BTreeSet::new();
    for spec in PUBLIC_OPERATOR_VARS {
        assert!(
            spec.name.starts_with("FLUX_"),
            "Tier-1 primary name must use FLUX_ prefix: {}",
            spec.name
        );
        assert!(
            !spec.name.ends_with('_'),
            "prefix fragment forbidden: {}",
            spec.name
        );
        assert!(!spec.summary.is_empty(), "empty summary for {}", spec.name);
        assert!(
            names.insert(spec.name),
            "duplicate PUBLIC_OPERATOR_VARS name: {}",
            spec.name
        );
        if let Some(alias) = spec.alias {
            assert!(
                alias.starts_with("ATLAS_"),
                "alias must use ATLAS_ prefix: {alias}"
            );
            assert!(
                aliases.insert(alias),
                "duplicate PUBLIC_OPERATOR_VARS alias: {alias}"
            );
            assert_eq!(
                flux_env_alias_key(spec.name).as_deref(),
                Some(alias),
                "alias for {} must agree with flux_env_alias_key",
                spec.name
            );
        }
        assert_eq!(public_var_spec(spec.name), Some(spec));
        if let Some(alias) = spec.alias {
            assert_eq!(public_var_spec(alias), Some(spec));
        }
    }
    assert_eq!(PUBLIC_OPERATOR_VARS.len(), 40);
}
