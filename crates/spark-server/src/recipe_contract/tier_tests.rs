// SPDX-License-Identifier: AGPL-3.0-only

//! Two-tier environment governance and catalog drift-guard tests.

use spark_server::flux_config::{
    PUBLIC_OPERATOR_VARS, PublicVarClass, PublicVarScope, flux_env_alias_key,
};

use super::catalog::{Catalog, EnvClass, EnvTier};

#[test]
fn builtin_catalog_tiers_and_flux_config_specs_stay_in_sync() {
    let c = Catalog::builtin().expect("builtin catalog");
    assert!(!c.is_empty());
    let (public, internal) = c.tier_counts();
    assert!(public > 0, "expected Tier-1 public operator entries");
    assert!(internal > 0, "expected Tier-2 internal runtime entries");
    assert_eq!(public + internal, c.len(), "tiers must partition catalog");

    // Verify every RuntimeEngine Tier-1 variable is present in the built-in
    // catalog (directly or via canonical alias), classified as PublicOperator,
    // and matches its declared governance EnvClass.
    for spec in PUBLIC_OPERATOR_VARS {
        // Alias parity with `main::flux_env_alias_key`.
        assert_eq!(
            crate::flux_env_alias_key(std::ffi::OsStr::new(spec.name)),
            flux_env_alias_key(spec.name),
            "alias mismatch between main.rs and flux_config for {}",
            spec.name
        );
        if spec.scope != PublicVarScope::RuntimeEngine {
            continue;
        }
        let canon = c.canonical(spec.name);
        let actual_class = c
            .class(spec.name)
            .or_else(|| c.class(&canon))
            .unwrap_or_else(|| {
                panic!(
                    "RuntimeEngine Tier-1 var {} (canonical {canon}) missing from env_catalog.tsv",
                    spec.name
                )
            });
        let expected_class = match spec.class {
            PublicVarClass::Numerics => EnvClass::Numerics,
            PublicVarClass::Route => EnvClass::Route,
            PublicVarClass::Resource => EnvClass::Resource,
            PublicVarClass::Observability => EnvClass::Observability,
            PublicVarClass::Debug => EnvClass::Debug,
        };
        assert_eq!(
            actual_class, expected_class,
            "EnvClass drift between FluxConfig and env_catalog.tsv for {} (canonical {canon})",
            spec.name
        );
        assert_eq!(
            c.tier(spec.name),
            Some(EnvTier::PublicOperator),
            "{} must resolve to Tier-1 PublicOperator",
            spec.name
        );
        if let Some(alias) = spec.alias {
            assert_eq!(
                c.tier(alias),
                Some(EnvTier::PublicOperator),
                "alias {alias} must resolve to Tier-1 PublicOperator"
            );
        }
    }

    // Special shorthand aliases map cleanly into Tier-1.
    assert_eq!(
        c.tier("FLUX_PREFILL_WIDE"),
        Some(EnvTier::PublicOperator),
        "FLUX_PREFILL_WIDE shorthand must resolve to Tier-1"
    );
    assert_eq!(
        c.tier("FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB"),
        Some(EnvTier::PublicOperator),
        "FP8 weight-cache reserve shorthand must resolve to Tier-1"
    );
    // Internal recipe/kernel knobs stay in Tier-2.
    assert_eq!(c.tier("ATLAS_MOE_T_LANES"), Some(EnvTier::InternalRuntime));
    assert_eq!(
        c.tier("ATLAS_QWEN4_PREFILL_GDN_FI"),
        Some(EnvTier::InternalRuntime)
    );
    assert_eq!(c.tier("ATLAS_DOES_NOT_EXIST_V11"), None);
}

#[test]
fn pruned_dead_and_false_positive_vars_stay_absent_from_catalog() {
    let c = Catalog::builtin().expect("builtin catalog");
    for absent in [
        // Confirmed-dead readers pruned from crates/
        "ATLAS_GDN_DUMP",
        "ATLAS_GDN_DUMP_LAYERS",
        "ATLAS_GDN_DUMP_N_SSM",
        // Stale catalog entries whose source readers were previously removed
        "ATLAS_DEBUG_SYNC_KERNELS",
        "ATLAS_QWEN4_SEQSTATE_RAM_MARGIN_GIB",
        "ATLAS_QWEN4_VERIFY_DUMP",
        "ATLAS_QWEN4_VERIFY_TOKEN_MAJOR",
        // Non-reader tracing event tags
        "ATLAS_PROFILE_RECEIPT",
        "ATLAS_PROFILE_LOOKUP_OVERLAY",
        "ATLAS_TS4E128_MOE_SCALES_READY",
        // Non-reader exclusion/prefix tables
        "ATLAS_DUMP",
        "ATLAS_LOG",
        "ATLAS_METRICS",
        "ATLAS_PORT",
        "ATLAS_HOST",
        "ATLAS_REQUEST_DUMP",
        "ATLAS_STARTUP_LEDGER",
        "ATLAS_TRACE",
        "FLUX_ACTIVE_ARGV_FILE",
        "FLUX_MODELS_ROOT",
        "FLUX_OPT_OUT",
        "FLUX_QUIET",
        // Test/bench/example-only literals
        "ATLAS_BENCH_M",
        "ATLAS_FI_CABI_LIB",
        "ATLAS_QWEN4_SCOPED_ENV_LOCK_TEST_UNUSED",
    ] {
        assert!(
            !c.has_reader(absent),
            "pruned dead/false-positive var must not reappear in env_catalog.tsv: {absent}"
        );
    }

    // Dynamic readers recovered by env_inventory.py remain tracked.
    for recovered in [
        "ATLAS_THINK_NUDGE",
        "FLUX_THINK_NUDGE",
        "ATLAS_SETTINGS_FILE",
        "FLUX_SETTINGS_FILE",
        "ATLAS_MEM_FLOOR_MARGIN_MIB",
        "FLUX_MEM_FLOOR_MARGIN_MIB",
        "ATLAS_LOG_TOOL_RAW",
        "FLUX_LOG_TOOL_RAW",
    ] {
        assert!(
            c.has_reader(recovered),
            "recovered production reader missing from env_catalog.tsv: {recovered}"
        );
    }
}

#[test]
fn custom_catalog_tier_counts_partition_accurately() {
    let custom = Catalog::parse(
        "ATLAS_MAX_BATCH_TOKENS\tRESOURCE\n\
         FLUX_DEBUG\tDEBUG\n\
         ATLAS_MOE_T_LANES\tROUTE\n\
         ATLAS_QWEN4_PREFILL_GDN_FI\tNUMERICS\n",
    )
    .expect("custom catalog");
    assert_eq!(custom.len(), 4);
    assert_eq!(custom.tier_counts(), (2, 2));
    assert_eq!(
        custom.tier("FLUX_MAX_BATCH_TOKENS"),
        Some(EnvTier::PublicOperator)
    );
    assert_eq!(
        custom.tier("ATLAS_MOE_T_LANES"),
        Some(EnvTier::InternalRuntime)
    );
    assert_eq!(custom.tier("ATLAS_UNKNOWN_KEY"), None);
}
