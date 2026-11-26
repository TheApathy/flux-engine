// SPDX-License-Identifier: AGPL-3.0-only

//! Typed Tier-1 Public Operator Configuration (`FluxConfig`).
//!
//! Flux Engine partitions its environment surface into two explicit tiers:
//!
//! - **Tier 1 (`PublicOperator`)** — the curated [`PUBLIC_OPERATOR_VARS`]
//!   surface parsed and validated by [`FluxConfig`]. Operators configure model
//!   selection, concurrency, KV cache budgets, speculation, PLE RAM staging,
//!   memory headroom, and HTTP/auth/logging through these `FLUX_*` variables.
//! - **Tier 2 (`InternalRuntime`)** — recipe-pinned kernel/route/precision
//!   engineering switches tracked by the fail-closed recipe contract
//!   (`recipe_contract::catalog::Catalog`).
//!
//! Alias precedence matches startup alias synchronization: when both a
//! `FLUX_*` variable and its canonical `ATLAS_*` alias are set in the process
//! environment, the `ATLAS_*` spelling wins.

mod spec;
mod types;

pub use spec::{
    PUBLIC_OPERATOR_VARS, PublicVarClass, PublicVarScope, PublicVarSpec, flux_env_alias_key,
    is_public_operator_var, public_var_spec,
};
pub use types::{
    DflashLookupMode, FluxConfigError, KvCacheDtype, LoadDebugMode, MemWatchMode, MtpSource,
    NativeBatchMode, SecretString, SettingsFileMode,
};

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::str::FromStr;

/// Typed representation of the Tier-1 Public Operator Configuration surface.
#[derive(Clone, Debug, PartialEq)]
pub struct FluxConfig {
    /// `FLUX_TARGET_HW` / `ATLAS_TARGET_HW` (default: `None` -> build default `gb10`).
    pub target_hw: Option<String>,
    /// `FLUX_TARGET_MODEL` / `ATLAS_TARGET_MODEL` (default: `None`).
    pub target_model: Option<String>,
    /// `FLUX_TARGET_QUANT` / `ATLAS_TARGET_QUANT` (default: `None`).
    pub target_quant: Option<String>,
    /// `FLUX_MODEL` (default: `None` -> auto-discover under `/models`).
    pub model: Option<PathBuf>,
    /// `FLUX_MODE` (default: `"auto"`).
    pub mode: String,
    /// `FLUX_RECIPE` (default: `None` -> legacy unpinned mode).
    pub recipe: Option<PathBuf>,
    /// `FLUX_WEIGHT_CACHE_DIR` / `ATLAS_WEIGHT_CACHE_DIR` (default: `None`).
    pub weight_cache_dir: Option<PathBuf>,
    /// `FLUX_MAX_SEQ_LEN` (default: `None` -> mode-dependent).
    pub max_seq_len: Option<usize>,
    /// `FLUX_CONCURRENCY` (default: `1`).
    pub concurrency: usize,
    /// `FLUX_MAX_BATCH_TOKENS` / `ATLAS_MAX_BATCH_TOKENS` (default: `None`).
    pub max_batch_tokens: Option<usize>,
    /// `FLUX_QWEN4_PREFILL_WIDE` / `ATLAS_QWEN4_PREFILL_WIDE` / `FLUX_PREFILL_WIDE`.
    pub qwen4_prefill_wide: Option<usize>,
    /// `FLUX_QWEN4_NATIVE_BATCH` / `ATLAS_QWEN4_NATIVE_BATCH` (default: `Auto`).
    pub qwen4_native_batch: NativeBatchMode,
    /// `FLUX_QUEUE_TIMEOUT` in seconds (default: `0`, disabled).
    pub queue_timeout_secs: u64,
    /// `FLUX_KV_CACHE_DTYPE` (default: `Auto`).
    pub kv_cache_dtype: KvCacheDtype,
    /// `FLUX_KV_POOL_TOKENS` (default: `None` -> auto-sized).
    pub kv_pool_tokens: Option<usize>,
    /// `FLUX_KV_MAX_TOKENS` (default: `None` -> uncapped by hard override).
    pub kv_max_tokens: Option<usize>,
    /// `FLUX_KV_POOL_UNCAPPED` (default: `false`).
    pub kv_pool_uncapped: bool,
    /// `FLUX_GPU_MEMORY_UTILIZATION` in `(0.0, 1.0]` (default: `0.55`).
    pub gpu_memory_utilization: f32,
    /// `FLUX_MEM_WATCH` / `ATLAS_MEM_WATCH` (default: `Auto`).
    pub mem_watch: MemWatchMode,
    /// `FLUX_MEM_FLOOR_MIB` / `ATLAS_MEM_FLOOR_MIB` (default: `6656`).
    pub mem_floor_mib: usize,
    /// `FLUX_MEMAVAILABLE_FLOOR_MIB` / `ATLAS_MEMAVAILABLE_FLOOR_MIB` (default: `None`).
    pub memavailable_floor_mib: Option<usize>,
    /// `FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB` / `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB`.
    pub fp8_weight_cache_reserve_gb: Option<usize>,
    /// `FLUX_DISABLE_SPECULATION` / `ATLAS_DISABLE_SPECULATION` (default: `false`).
    pub disable_speculation: bool,
    /// `FLUX_DRAFT_MODEL` (default: `None` -> auto-detect).
    pub draft_model: Option<PathBuf>,
    /// `FLUX_GAMMA` (default: `None` -> drafter default).
    pub gamma: Option<usize>,
    /// `FLUX_MTP_SOURCE` / `ATLAS_MTP_SOURCE` (default: `Auto`).
    pub mtp_source: MtpSource,
    /// `FLUX_DFLASH_DRAFT_CAP` / `ATLAS_DFLASH_DRAFT_CAP` (default: `None`).
    pub dflash_draft_cap: Option<usize>,
    /// `FLUX_DFLASH_LOOKUP` / `ATLAS_DFLASH_LOOKUP` (default: `Off`).
    pub dflash_lookup: DflashLookupMode,
    /// `FLUX_PLE_MMAP` / `ATLAS_PLE_MMAP` (default: `false`).
    pub ple_mmap: bool,
    /// `FLUX_PLE_CACHE_MB` / `ATLAS_PLE_CACHE_MB` (default: `None`).
    pub ple_cache_mb: Option<usize>,
    /// `FLUX_PLE_PARKED_CACHE_MB` / `ATLAS_PLE_PARKED_CACHE_MB` (default: `None`).
    pub ple_parked_cache_mb: Option<usize>,
    /// `FLUX_HOST` (default: `"127.0.0.1"`).
    pub host: String,
    /// `FLUX_PORT` (default: `8899`).
    pub port: u16,
    /// `FLUX_API_KEY` / `ATLAS_API_KEY` (default: `None`; redacted in `Debug`).
    pub api_key: Option<SecretString>,
    /// `FLUX_TRUST_PROXY` / `ATLAS_TRUST_PROXY` (default: `false`).
    pub trust_proxy: bool,
    /// `FLUX_SETTINGS_FILE` / `ATLAS_SETTINGS_FILE` (default: `Off`).
    pub settings_file: SettingsFileMode,
    /// `FLUX_HOME` / `ATLAS_HOME` (default: `None`).
    pub home: Option<PathBuf>,
    /// `FLUX_DEBUG` (default: `false`).
    pub debug: bool,
    /// `FLUX_LOAD_DEBUG` (default: `Auto`).
    pub load_debug: LoadDebugMode,
    /// `FLUX_PROFILE` / `ATLAS_PROFILE` (default: `false`).
    pub profile: bool,
}

impl Default for FluxConfig {
    fn default() -> Self {
        Self {
            target_hw: None,
            target_model: None,
            target_quant: None,
            model: None,
            mode: "auto".to_string(),
            recipe: None,
            weight_cache_dir: None,
            max_seq_len: None,
            concurrency: 1,
            max_batch_tokens: None,
            qwen4_prefill_wide: None,
            qwen4_native_batch: NativeBatchMode::Auto,
            queue_timeout_secs: 0,
            kv_cache_dtype: KvCacheDtype::Auto,
            kv_pool_tokens: None,
            kv_max_tokens: None,
            kv_pool_uncapped: false,
            gpu_memory_utilization: 0.55,
            mem_watch: MemWatchMode::Auto,
            mem_floor_mib: 6_656,
            memavailable_floor_mib: None,
            fp8_weight_cache_reserve_gb: None,
            disable_speculation: false,
            draft_model: None,
            gamma: None,
            mtp_source: MtpSource::Auto,
            dflash_draft_cap: None,
            dflash_lookup: DflashLookupMode::Off,
            ple_mmap: false,
            ple_cache_mb: None,
            ple_parked_cache_mb: None,
            host: "127.0.0.1".to_string(),
            port: 8899,
            api_key: None,
            trust_proxy: false,
            settings_file: SettingsFileMode::Off,
            home: None,
            debug: false,
            load_debug: LoadDebugMode::Auto,
            profile: false,
        }
    }
}

impl FluxConfig {
    /// Parse and validate [`FluxConfig`] from the current process environment.
    ///
    /// # Errors
    ///
    /// Returns [`FluxConfigError`] if any Tier-1 variable fails parsing or
    /// violates range or cross-field validation rules.
    pub fn from_env() -> Result<Self, FluxConfigError> {
        let map: BTreeMap<String, String> = std::env::vars()
            .filter(|(k, _)| k.starts_with("FLUX_") || k.starts_with("ATLAS_"))
            .collect();
        Self::from_map(&map)
    }

    /// Parse and validate [`FluxConfig`] deterministically from an explicit key-value map.
    ///
    /// Alias precedence matches startup alias sync: when both an `ATLAS_*` alias
    /// and a `FLUX_*` spelling are present, the `ATLAS_*` value wins.
    ///
    /// # Errors
    ///
    /// Returns [`FluxConfigError`] if any present variable is malformed, out of
    /// bounds, or mutually inconsistent.
    pub fn from_map(vars: &BTreeMap<String, String>) -> Result<Self, FluxConfigError> {
        let mut cfg = Self {
            target_hw: lookup_nonempty(vars, &["ATLAS_TARGET_HW", "FLUX_TARGET_HW"]),
            target_model: lookup_nonempty(vars, &["ATLAS_TARGET_MODEL", "FLUX_TARGET_MODEL"]),
            target_quant: lookup_nonempty(vars, &["ATLAS_TARGET_QUANT", "FLUX_TARGET_QUANT"]),
            model: lookup_nonempty(vars, &["FLUX_MODEL"]).map(PathBuf::from),
            recipe: lookup_nonempty(vars, &["FLUX_RECIPE"]).map(PathBuf::from),
            weight_cache_dir: lookup_nonempty(
                vars,
                &["ATLAS_WEIGHT_CACHE_DIR", "FLUX_WEIGHT_CACHE_DIR"],
            )
            .map(PathBuf::from),
            max_seq_len: parse_opt_int(vars, "FLUX_MAX_SEQ_LEN", &["FLUX_MAX_SEQ_LEN"])?,
            max_batch_tokens: parse_opt_int(
                vars,
                "FLUX_MAX_BATCH_TOKENS",
                &["ATLAS_MAX_BATCH_TOKENS", "FLUX_MAX_BATCH_TOKENS"],
            )?,
            qwen4_prefill_wide: parse_opt_int(
                vars,
                "FLUX_QWEN4_PREFILL_WIDE",
                &[
                    "ATLAS_QWEN4_PREFILL_WIDE",
                    "FLUX_QWEN4_PREFILL_WIDE",
                    "FLUX_PREFILL_WIDE",
                ],
            )?,
            kv_pool_tokens: parse_opt_int(vars, "FLUX_KV_POOL_TOKENS", &["FLUX_KV_POOL_TOKENS"])?,
            kv_max_tokens: parse_opt_int(vars, "FLUX_KV_MAX_TOKENS", &["FLUX_KV_MAX_TOKENS"])?,
            kv_pool_uncapped: parse_bool(
                vars,
                "FLUX_KV_POOL_UNCAPPED",
                &["FLUX_KV_POOL_UNCAPPED"],
            )?,
            memavailable_floor_mib: parse_opt_int(
                vars,
                "FLUX_MEMAVAILABLE_FLOOR_MIB",
                &[
                    "ATLAS_MEMAVAILABLE_FLOOR_MIB",
                    "FLUX_MEMAVAILABLE_FLOOR_MIB",
                ],
            )?,
            fp8_weight_cache_reserve_gb: parse_opt_int(
                vars,
                "FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB",
                &[
                    "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB",
                    "FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB",
                ],
            )?,
            disable_speculation: parse_bool(
                vars,
                "FLUX_DISABLE_SPECULATION",
                &["ATLAS_DISABLE_SPECULATION", "FLUX_DISABLE_SPECULATION"],
            )?,
            draft_model: lookup_nonempty(vars, &["FLUX_DRAFT_MODEL"]).map(PathBuf::from),
            gamma: parse_opt_int(vars, "FLUX_GAMMA", &["FLUX_GAMMA"])?,
            dflash_draft_cap: parse_opt_int(
                vars,
                "FLUX_DFLASH_DRAFT_CAP",
                &["ATLAS_DFLASH_DRAFT_CAP", "FLUX_DFLASH_DRAFT_CAP"],
            )?,
            ple_mmap: parse_bool(vars, "FLUX_PLE_MMAP", &["ATLAS_PLE_MMAP", "FLUX_PLE_MMAP"])?,
            ple_cache_mb: parse_opt_int(
                vars,
                "FLUX_PLE_CACHE_MB",
                &["ATLAS_PLE_CACHE_MB", "FLUX_PLE_CACHE_MB"],
            )?,
            ple_parked_cache_mb: parse_opt_int(
                vars,
                "FLUX_PLE_PARKED_CACHE_MB",
                &["ATLAS_PLE_PARKED_CACHE_MB", "FLUX_PLE_PARKED_CACHE_MB"],
            )?,
            api_key: lookup_nonempty(vars, &["ATLAS_API_KEY", "FLUX_API_KEY"])
                .map(SecretString::new),
            trust_proxy: parse_bool(
                vars,
                "FLUX_TRUST_PROXY",
                &["ATLAS_TRUST_PROXY", "FLUX_TRUST_PROXY"],
            )?,
            home: lookup_nonempty(vars, &["ATLAS_HOME", "FLUX_HOME"]).map(PathBuf::from),
            debug: parse_bool(vars, "FLUX_DEBUG", &["FLUX_DEBUG"])?,
            profile: parse_bool(vars, "FLUX_PROFILE", &["ATLAS_PROFILE", "FLUX_PROFILE"])?,
            ..Self::default()
        };
        if let Some(mode) = lookup_nonempty(vars, &["FLUX_MODE"]) {
            cfg.mode = mode;
        }
        if let Some(c) = parse_opt_int(vars, "FLUX_CONCURRENCY", &["FLUX_CONCURRENCY"])? {
            cfg.concurrency = c;
        }
        if let Some(nb) = parse_opt_enum(
            vars,
            "FLUX_QWEN4_NATIVE_BATCH",
            &["ATLAS_QWEN4_NATIVE_BATCH", "FLUX_QWEN4_NATIVE_BATCH"],
        )? {
            cfg.qwen4_native_batch = nb;
        }
        if let Some(t) = parse_opt_int(vars, "FLUX_QUEUE_TIMEOUT", &["FLUX_QUEUE_TIMEOUT"])? {
            cfg.queue_timeout_secs = t;
        }
        if let Some(dt) = parse_opt_enum(vars, "FLUX_KV_CACHE_DTYPE", &["FLUX_KV_CACHE_DTYPE"])? {
            cfg.kv_cache_dtype = dt;
        }
        if let Some(util) = parse_opt_f32(
            vars,
            "FLUX_GPU_MEMORY_UTILIZATION",
            &["FLUX_GPU_MEMORY_UTILIZATION"],
        )? {
            cfg.gpu_memory_utilization = util;
        }
        if let Some(mw) = parse_opt_enum(
            vars,
            "FLUX_MEM_WATCH",
            &["ATLAS_MEM_WATCH", "FLUX_MEM_WATCH"],
        )? {
            cfg.mem_watch = mw;
        }
        if let Some(floor) = parse_opt_int(
            vars,
            "FLUX_MEM_FLOOR_MIB",
            &["ATLAS_MEM_FLOOR_MIB", "FLUX_MEM_FLOOR_MIB"],
        )? {
            cfg.mem_floor_mib = floor;
        }
        if let Some(src) = parse_opt_enum(
            vars,
            "FLUX_MTP_SOURCE",
            &["ATLAS_MTP_SOURCE", "FLUX_MTP_SOURCE"],
        )? {
            cfg.mtp_source = src;
        }
        if let Some(lk) = parse_opt_enum(
            vars,
            "FLUX_DFLASH_LOOKUP",
            &["ATLAS_DFLASH_LOOKUP", "FLUX_DFLASH_LOOKUP"],
        )? {
            cfg.dflash_lookup = lk;
        }
        if let Some(host) = lookup_nonempty(vars, &["FLUX_HOST"]) {
            cfg.host = host;
        }
        if let Some(port) = parse_opt_int(vars, "FLUX_PORT", &["FLUX_PORT"])? {
            cfg.port = port;
        }
        if let Some(raw) = lookup_raw(vars, &["ATLAS_SETTINGS_FILE", "FLUX_SETTINGS_FILE"]) {
            cfg.settings_file = SettingsFileMode::parse(raw);
        }
        if let Some(ld) = parse_opt_enum(vars, "FLUX_LOAD_DEBUG", &["FLUX_LOAD_DEBUG"])? {
            cfg.load_debug = ld;
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate domain bounds and cross-field invariants.
    ///
    /// # Errors
    ///
    /// Returns [`FluxConfigError::OutOfRange`] or [`FluxConfigError::Conflict`]
    /// when any field violates its contract.
    pub fn validate(&self) -> Result<(), FluxConfigError> {
        require_positive(self.max_seq_len, "FLUX_MAX_SEQ_LEN")?;
        require_positive(Some(self.concurrency), "FLUX_CONCURRENCY")?;
        require_positive(self.max_batch_tokens, "FLUX_MAX_BATCH_TOKENS")?;
        require_positive(self.qwen4_prefill_wide, "FLUX_QWEN4_PREFILL_WIDE")?;
        require_positive(self.kv_pool_tokens, "FLUX_KV_POOL_TOKENS")?;
        require_positive(self.kv_max_tokens, "FLUX_KV_MAX_TOKENS")?;
        require_positive(self.gamma, "FLUX_GAMMA")?;
        require_positive(self.dflash_draft_cap, "FLUX_DFLASH_DRAFT_CAP")?;
        if !self.gpu_memory_utilization.is_finite()
            || self.gpu_memory_utilization <= 0.0
            || self.gpu_memory_utilization > 1.0
        {
            return Err(FluxConfigError::OutOfRange {
                key: "FLUX_GPU_MEMORY_UTILIZATION",
                raw: self.gpu_memory_utilization.to_string(),
                constraint: "must be a finite float in (0.0, 1.0]",
            });
        }
        if self.port == 0 {
            return Err(FluxConfigError::OutOfRange {
                key: "FLUX_PORT",
                raw: "0".to_string(),
                constraint: "must be in 1..=65535",
            });
        }
        if let (Some(pool), Some(max)) = (self.kv_pool_tokens, self.kv_max_tokens)
            && pool > max
        {
            return Err(FluxConfigError::Conflict {
                message: format!("FLUX_KV_POOL_TOKENS ({pool}) exceeds FLUX_KV_MAX_TOKENS ({max})"),
            });
        }
        if self.qwen4_native_batch == NativeBatchMode::ForceNative && self.concurrency > 32 {
            return Err(FluxConfigError::Conflict {
                message: format!(
                    "FLUX_QWEN4_NATIVE_BATCH=1 supports FLUX_CONCURRENCY <= 32, got {}",
                    self.concurrency
                ),
            });
        }
        Ok(())
    }
}

fn require_positive(val: Option<usize>, key: &'static str) -> Result<(), FluxConfigError> {
    if val == Some(0) {
        return Err(FluxConfigError::OutOfRange {
            key,
            raw: "0".to_string(),
            constraint: "must be strictly positive (>= 1)",
        });
    }
    Ok(())
}

fn lookup_raw<'a>(vars: &'a BTreeMap<String, String>, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| vars.get(*k).map(String::as_str))
}

fn lookup_nonempty(vars: &BTreeMap<String, String>, keys: &[&str]) -> Option<String> {
    lookup_raw(vars, keys)
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string)
}

fn parse_opt_int<T: FromStr>(
    vars: &BTreeMap<String, String>,
    key: &'static str,
    keys: &[&str],
) -> Result<Option<T>, FluxConfigError> {
    let Some(raw) = lookup_nonempty(vars, keys) else {
        return Ok(None);
    };
    raw.parse::<T>()
        .map(Some)
        .map_err(|_| FluxConfigError::InvalidValue {
            key,
            raw,
            expected: "non-negative integer",
        })
}

fn parse_opt_f32(
    vars: &BTreeMap<String, String>,
    key: &'static str,
    keys: &[&str],
) -> Result<Option<f32>, FluxConfigError> {
    let Some(raw) = lookup_nonempty(vars, keys) else {
        return Ok(None);
    };
    raw.parse::<f32>()
        .map(Some)
        .map_err(|_| FluxConfigError::InvalidValue {
            key,
            raw,
            expected: "floating-point number",
        })
}

fn parse_bool(
    vars: &BTreeMap<String, String>,
    key: &'static str,
    keys: &[&str],
) -> Result<bool, FluxConfigError> {
    let Some(raw) = lookup_nonempty(vars, keys) else {
        return Ok(false);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Ok(true),
        "0" | "false" | "off" | "no" => Ok(false),
        _ => Err(FluxConfigError::InvalidValue {
            key,
            raw,
            expected: "boolean (1|0|true|false|on|off)",
        }),
    }
}

fn parse_opt_enum<T: FromStr<Err = &'static str>>(
    vars: &BTreeMap<String, String>,
    key: &'static str,
    keys: &[&str],
) -> Result<Option<T>, FluxConfigError> {
    let Some(raw) = lookup_nonempty(vars, keys) else {
        return Ok(None);
    };
    raw.parse::<T>()
        .map(Some)
        .map_err(|expected| FluxConfigError::InvalidValue { key, raw, expected })
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<FluxConfig>();
    assert_send_sync::<FluxConfigError>();
};

#[cfg(test)]
mod tests;
