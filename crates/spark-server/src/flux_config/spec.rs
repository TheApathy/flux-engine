// SPDX-License-Identifier: AGPL-3.0-only

//! Tier-1 Public Operator Environment Variable specification table and alias lookup.

/// Semantic governance class for a Tier-1 public operator variable.
///
/// Mirrors `recipe_contract::catalog::EnvClass` so the library target stays
/// independent of the binary-only CLI/recipe modules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PublicVarClass {
    /// Changes served output, precision, or kernel arithmetic.
    Numerics,
    /// Selects a kernel or scheduling route claiming identical served output.
    Route,
    /// Configures memory, threads, limits, paths, network, auth, or I/O shape.
    Resource,
    /// Controls metrics, profiling, receipts, or structured logging.
    Observability,
    /// Diagnostic or debug-log controls.
    Debug,
}

impl PublicVarClass {
    /// Canonical uppercase TSV label matching `env_catalog.tsv`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Numerics => "NUMERICS",
            Self::Route => "ROUTE",
            Self::Resource => "RESOURCE",
            Self::Observability => "OBSERVABILITY",
            Self::Debug => "DEBUG",
        }
    }
}

/// Where a Tier-1 public operator variable is consumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PublicVarScope {
    /// Read directly by the compiled server/model/storage runtime (cataloged in `env_catalog.tsv`).
    RuntimeEngine,
    /// Consumed by the `flux` launcher / container entrypoint and translated into CLI flags.
    Launcher,
    /// Consumed by Cargo `build.rs` scripts at compile time.
    BuildTime,
    /// Selects the fail-closed `recipe.toml` contract before catalog evaluation.
    ContractControl,
}

/// Metadata entry describing one Tier-1 public operator environment variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicVarSpec {
    /// Canonical operator-facing `FLUX_*` environment variable name.
    pub name: &'static str,
    /// Underlying `ATLAS_*` alias read by the engine, if any.
    pub alias: Option<&'static str>,
    /// Governance classification.
    pub class: PublicVarClass,
    /// Consumption layer (runtime engine, launcher, build-time, or contract control).
    pub scope: PublicVarScope,
    /// Human-readable default value when unset.
    pub default: &'static str,
    /// Concise operator-facing summary.
    pub summary: &'static str,
}

const fn spec(
    name: &'static str,
    alias: Option<&'static str>,
    class: PublicVarClass,
    scope: PublicVarScope,
    default: &'static str,
    summary: &'static str,
) -> PublicVarSpec {
    PublicVarSpec {
        name,
        alias,
        class,
        scope,
        default,
        summary,
    }
}

use PublicVarClass::{Debug as ClsDebug, Numerics, Observability, Resource, Route};
use PublicVarScope::{BuildTime, ContractControl, Launcher, RuntimeEngine};

/// Single source of truth for the Tier-1 Public Operator Environment Surface (40 variables).
pub const PUBLIC_OPERATOR_VARS: &[PublicVarSpec] = &[
    // ── Target hardware, model, mode & contract ──────────────────────────
    spec(
        "FLUX_TARGET_HW",
        Some("ATLAS_TARGET_HW"),
        Resource,
        BuildTime,
        "gb10",
        "Compile-time target GPU/backend architecture (gb10, rtx6000, sm120, metal)",
    ),
    spec(
        "FLUX_TARGET_MODEL",
        Some("ATLAS_TARGET_MODEL"),
        Resource,
        RuntimeEngine,
        "unset",
        "Target kernel model family selector and transform-cache key",
    ),
    spec(
        "FLUX_TARGET_QUANT",
        Some("ATLAS_TARGET_QUANT"),
        Resource,
        RuntimeEngine,
        "unset",
        "Target weight quantization family (nvfp4, fp8, bf16, exl3)",
    ),
    spec(
        "FLUX_MODEL",
        None,
        Resource,
        Launcher,
        "auto",
        "Path to the local model checkpoint directory",
    ),
    spec(
        "FLUX_MODE",
        None,
        Observability,
        RuntimeEngine,
        "auto",
        "Named operating profile (auto, max-prefill, interactive, fast-prefill, decode, long-context, exact)",
    ),
    spec(
        "FLUX_RECIPE",
        None,
        Resource,
        ContractControl,
        "unset",
        "Path to a fail-closed recipe.toml contract enforced at startup",
    ),
    spec(
        "FLUX_WEIGHT_CACHE_DIR",
        Some("ATLAS_WEIGHT_CACHE_DIR"),
        Resource,
        RuntimeEngine,
        "auto",
        "Directory for persistent post-transform weight cache shards",
    ),
    // ── Concurrency, batching & sequence limits ──────────────────────────
    spec(
        "FLUX_MAX_SEQ_LEN",
        None,
        Resource,
        Launcher,
        "mode-dependent",
        "Maximum context length in tokens (--max-seq-len)",
    ),
    spec(
        "FLUX_CONCURRENCY",
        None,
        Resource,
        Launcher,
        "1",
        "Maximum concurrent sequences (--max-batch-size / --max-num-seqs)",
    ),
    spec(
        "FLUX_MAX_BATCH_TOKENS",
        Some("ATLAS_MAX_BATCH_TOKENS"),
        Resource,
        RuntimeEngine,
        "unset",
        "Maximum combined tokens scheduled in a single forward batch",
    ),
    spec(
        "FLUX_QWEN4_PREFILL_WIDE",
        Some("ATLAS_QWEN4_PREFILL_WIDE"),
        Route,
        RuntimeEngine,
        "unset",
        "Prefill chunk width in tokens (4096, 8192, 16384; shorthand FLUX_PREFILL_WIDE)",
    ),
    spec(
        "FLUX_QWEN4_NATIVE_BATCH",
        Some("ATLAS_QWEN4_NATIVE_BATCH"),
        Route,
        RuntimeEngine,
        "auto",
        "Qwen4 C2-C32 native batched decode mode (auto, 1=require native, 0=serial)",
    ),
    spec(
        "FLUX_QUEUE_TIMEOUT",
        None,
        Resource,
        RuntimeEngine,
        "0",
        "Maximum seconds a request may wait in the admission queue (0 = disabled)",
    ),
    // ── KV cache & VRAM/UMA memory headroom ──────────────────────────────
    spec(
        "FLUX_KV_CACHE_DTYPE",
        None,
        Numerics,
        Launcher,
        "auto",
        "KV cache precision (auto, bf16, fp8, turboquant, nvfp4)",
    ),
    spec(
        "FLUX_KV_POOL_TOKENS",
        None,
        Resource,
        Launcher,
        "auto",
        "Explicit KV pool token capacity passed to --kv-pool-tokens",
    ),
    spec(
        "FLUX_KV_MAX_TOKENS",
        None,
        Resource,
        RuntimeEngine,
        "unset",
        "Hard runtime ceiling on allocated KV pool tokens",
    ),
    spec(
        "FLUX_KV_POOL_UNCAPPED",
        None,
        Resource,
        RuntimeEngine,
        "0",
        "Allow KV pool sizing to bypass the default conservative token cap",
    ),
    spec(
        "FLUX_GPU_MEMORY_UTILIZATION",
        None,
        Resource,
        Launcher,
        "0.55",
        "Fraction of GPU/UMA memory budgeted for runtime arenas (<=0.55 on GB10 UMA)",
    ),
    spec(
        "FLUX_MEM_WATCH",
        Some("ATLAS_MEM_WATCH"),
        Resource,
        RuntimeEngine,
        "auto",
        "In-process MemAvailable floor watchdog (auto, on, off)",
    ),
    spec(
        "FLUX_MEM_FLOOR_MIB",
        Some("ATLAS_MEM_FLOOR_MIB"),
        Resource,
        RuntimeEngine,
        "6656",
        "MemAvailable floor in MiB before watchdog load-abort or serve-503 refusal",
    ),
    spec(
        "FLUX_MEMAVAILABLE_FLOOR_MIB",
        Some("ATLAS_MEMAVAILABLE_FLOOR_MIB"),
        Resource,
        RuntimeEngine,
        "auto",
        "Startup memory-admission free-RAM floor in MiB",
    ),
    spec(
        "FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB",
        Some("ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB"),
        Resource,
        RuntimeEngine,
        "7",
        "Unified-memory reserve in GiB below which FP8 weight caching stops allocating",
    ),
    // ── Speculative decoding (MTP & DFlash2) ─────────────────────────────
    spec(
        "FLUX_DISABLE_SPECULATION",
        Some("ATLAS_DISABLE_SPECULATION"),
        Route,
        RuntimeEngine,
        "0",
        "Disable all speculative decoding (MTP, DFlash, self/n-gram)",
    ),
    spec(
        "FLUX_DRAFT_MODEL",
        None,
        Resource,
        Launcher,
        "auto",
        "Path to companion speculative drafter directory (or none to disable sidecars)",
    ),
    spec(
        "FLUX_GAMMA",
        None,
        Route,
        Launcher,
        "auto",
        "Speculative draft tokens proposed per step (4 for MTP K=5, block_size-1 for DFlash)",
    ),
    spec(
        "FLUX_MTP_SOURCE",
        Some("ATLAS_MTP_SOURCE"),
        Route,
        RuntimeEngine,
        "auto",
        "MTP routed-expert weight source (auto, checkpoint, sidecar)",
    ),
    spec(
        "FLUX_DFLASH_DRAFT_CAP",
        Some("ATLAS_DFLASH_DRAFT_CAP"),
        Route,
        RuntimeEngine,
        "unset",
        "Hard upper bound on DFlash proposed draft tokens per step",
    ),
    spec(
        "FLUX_DFLASH_LOOKUP",
        Some("ATLAS_DFLASH_LOOKUP"),
        Route,
        RuntimeEngine,
        "off",
        "Opt-in DFlash lookup drafter mode (off, sam, retrieval)",
    ),
    // ── PLE RAM staging & offload ────────────────────────────────────────
    spec(
        "FLUX_PLE_MMAP",
        Some("ATLAS_PLE_MMAP"),
        Resource,
        RuntimeEngine,
        "0",
        "Memory-map PLE offload shards instead of O_DIRECT asynchronous reads",
    ),
    spec(
        "FLUX_PLE_CACHE_MB",
        Some("ATLAS_PLE_CACHE_MB"),
        Resource,
        RuntimeEngine,
        "auto",
        "Host RAM staging cache budget in MiB for active PLE embeddings",
    ),
    spec(
        "FLUX_PLE_PARKED_CACHE_MB",
        Some("ATLAS_PLE_PARKED_CACHE_MB"),
        Resource,
        RuntimeEngine,
        "auto",
        "Host RAM cache budget in MiB for parked PLE offload shards",
    ),
    // ── Network, auth, state & observability ─────────────────────────────
    spec(
        "FLUX_HOST",
        None,
        Resource,
        Launcher,
        "127.0.0.1",
        "HTTP API server bind host address",
    ),
    spec(
        "FLUX_PORT",
        None,
        Resource,
        Launcher,
        "8899",
        "HTTP API server listening TCP port",
    ),
    spec(
        "FLUX_API_KEY",
        Some("ATLAS_API_KEY"),
        Resource,
        RuntimeEngine,
        "unset",
        "Bearer API key activating HTTP authentication (redacted in Debug output)",
    ),
    spec(
        "FLUX_TRUST_PROXY",
        Some("ATLAS_TRUST_PROXY"),
        Resource,
        RuntimeEngine,
        "0",
        "Trust X-Forwarded-For / X-Real-IP headers for per-client rate limiting",
    ),
    spec(
        "FLUX_SETTINGS_FILE",
        Some("ATLAS_SETTINGS_FILE"),
        Resource,
        RuntimeEngine,
        "off",
        "Persistent runtime settings JSON file (off, on, or explicit path)",
    ),
    spec(
        "FLUX_HOME",
        Some("ATLAS_HOME"),
        Resource,
        RuntimeEngine,
        "auto",
        "Base state and configuration directory for Flux Engine",
    ),
    spec(
        "FLUX_DEBUG",
        None,
        ClsDebug,
        RuntimeEngine,
        "0",
        "Enable structured rolling debug log file (FLUX_DEBUG_FILE / _MAX_MB / _FILTER)",
    ),
    spec(
        "FLUX_LOAD_DEBUG",
        None,
        ClsDebug,
        RuntimeEngine,
        "auto",
        "Load-phase diagnostic capture mode (auto, always, off)",
    ),
    spec(
        "FLUX_PROFILE",
        Some("ATLAS_PROFILE"),
        Observability,
        RuntimeEngine,
        "0",
        "Enable per-step latency and CUDA kernel profiling instrumentation",
    ),
];

/// Map a `FLUX_*` environment key to its canonical `ATLAS_*` engine alias.
///
/// Mirrors `main::flux_env_alias_key` verbatim: launcher-only meta keys and
/// `FLUX_*`-native debug/log keys return `None`; shorthand aliases
/// (`FLUX_PREFILL_WIDE`, `FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB`)
/// map to their expanded `ATLAS_*` engine names.
#[must_use]
pub fn flux_env_alias_key(key: &str) -> Option<String> {
    const META_KEYS: &[&str] = &[
        "FLUX_MODE",
        "FLUX_QUIET",
        "FLUX_MODELS_ROOT",
        "FLUX_ATLAS_ROOT",
        "FLUX_ACTIVE_ARGV_FILE",
        "FLUX_OPT_OUT",
        "FLUX_THINKING",
        "FLUX_LOAD_DEBUG",
        "FLUX_LOAD_DEBUG_FILE",
        "FLUX_DEBUG",
        "FLUX_DEBUG_FILE",
        "FLUX_DEBUG_MAX_MB",
        "FLUX_DEBUG_FILTER",
    ];
    let suffix = key.strip_prefix("FLUX_")?;
    if META_KEYS.contains(&key) || suffix.is_empty() {
        return None;
    }
    Some(match suffix {
        "QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB" => {
            "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB".to_string()
        }
        "PREFILL_WIDE" => "ATLAS_QWEN4_PREFILL_WIDE".to_string(),
        _ => format!("ATLAS_{suffix}"),
    })
}

/// Look up the Tier-1 [`PublicVarSpec`] for either a `FLUX_*` name or its `ATLAS_*` alias.
#[must_use]
pub fn public_var_spec(name: &str) -> Option<&'static PublicVarSpec> {
    if let Some(spec) = PUBLIC_OPERATOR_VARS
        .iter()
        .find(|s| s.name == name || s.alias == Some(name))
    {
        return Some(spec);
    }
    let alias = flux_env_alias_key(name)?;
    PUBLIC_OPERATOR_VARS
        .iter()
        .find(|s| s.alias == Some(alias.as_str()))
}

/// Returns `true` if `name` (in either `FLUX_*` or `ATLAS_*` spelling) belongs to Tier 1.
#[must_use]
pub fn is_public_operator_var(name: &str) -> bool {
    public_var_spec(name).is_some()
}
