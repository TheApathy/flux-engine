// SPDX-License-Identifier: AGPL-3.0-only

//! GateSight recurrent-memory-guided KV eviction engine for hybrid GDN + FullAttention models.
//!
//! Implements both actionable deployment paths from arXiv:2610.14754 mapped onto Flux Engine's
//! `block_size = 16` NVFP4/BitDecoding-FP4 KV layout:
//! - **Path A (zero-training)**: load-time static slow-head classifier from `ssm.a_log` /
//!   `dt_bias` + in-register `(beta, g, ||v||_2)` rolling ring + max-pooled `block_size = 16`
//!   eviction (zero re-quantization for both `K` and transposed `V^T`).
//! - **Path B (full GateSight)**: lightweight `867 -> 64 -> 1` per-`(attn_layer, kv_head)`
//!   authenticated MLP sidecar + cross-layer output-gate budget allocation + streaming reservoir
//!   compaction with 2-zone `RadixTree` prefix protection.

pub mod planner;
pub mod scorer;
pub mod signals;

#[cfg(test)]
mod tests;

use std::fmt;
use std::path::PathBuf;

pub use planner::{
    CompactionPlan, CompactionRequest, LayerBudget, ModelFamilyPolicy, ScoredItem,
    StreamingTopKReservoir, allocate_layer_budgets, is_bitfp4_block_aligned, plan_zone2_compaction,
};
pub use scorer::{
    DecayWeighting, GATESIGHT_SIDECAR_MAGIC, GateSightMlpScorer, GateSightMlpSidecar,
    HeadMlpWeights, MlpActivation, ZeroTrainingBlockScorer, build_token_features,
};
pub use signals::{
    GdnLayerSlowHeads, GdnSignalRingDescriptor, GdnSignalRingHost, GdnSignalSample,
    ModelSlowHeadTable, static_half_life_tokens,
};

/// Atomic KV block granularity shared with BitDecoding-FP4 (`NVFP4_GROUP_SIZE = 16`).
pub const GATESIGHT_BLOCK_SIZE: usize = 16;

/// Number of preceding GDN layers aggregated per FullAttention layer in Qwen3.8 hybrids.
pub const DEFAULT_GDN_WINDOW_LAYERS: usize = 6;

/// Number of scalar GateSight signals per `(gdn_layer, token, v_head)`: `(beta, g, ||v||_2)`.
pub const GATESIGHT_SIGNALS_PER_HEAD: usize = 3;

/// Number of relative position / age features appended in GateSight Eq. (3).
pub const GATESIGHT_AGE_FEATURES: usize = 3;

/// Default half-life threshold (tokens) identifying slow GDN memory heads (GateSight §2.1).
pub const DEFAULT_SLOW_HEAD_HALF_LIFE_TOKENS: f32 = 1000.0;

/// Conservative minimum retention floor enforced on Qwen3.8-Flash-Next QSA to avoid double-pruning cliffs.
pub const FLASH_NEXT_QSA_MIN_RETENTION: f32 = 0.60;

/// Minimum retention ratio permitted on dense-GQA models (10x compression ceiling from Table 3).
pub const DENSE_GQA_MIN_RETENTION: f32 = 0.10;

/// Default L2 cache capacity on GB10 Spark (32 MiB per `docs/FLUX_KERNELS.md`).
pub const GB10_L2_CACHE_BYTES: usize = 32 * 1024 * 1024;

/// Default rolling-ring chunk length (8192 tokens -> 13.5 MiB FP16 for 6 GDN layers x 48 v-heads).
pub const DEFAULT_RING_CHUNK_TOKENS: usize = 8192;

/// Environment variable selecting opt-in GateSight KV eviction mode (default OFF).
pub const GATESIGHT_EVICT_ENV: &str = "ATLAS_GATESIGHT_EVICT";

/// Environment variable overriding target Zone-2 KV retention ratio `p in (0, 1]`.
pub const GATESIGHT_RETENTION_ENV: &str = "ATLAS_GATESIGHT_RETENTION";

/// Environment variable pointing to an authenticated `ATLGTS01` Path-B MLP sidecar file.
pub const GATESIGHT_SIDECAR_PATH_ENV: &str = "ATLAS_GATESIGHT_SIDECAR_PATH";

/// Errors produced by GateSight configuration, sidecar parsing, scoring, or planning.
#[derive(Debug, Clone, PartialEq)]
pub enum GateSightError {
    /// Invalid environment variable or configuration parameter.
    InvalidConfig(String),
    /// Dimension mismatch between signal tensors, sidecar weights, and layer config.
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        actual: usize,
    },
    /// Sidecar binary header, version, length, or SHA-256 digest failed validation.
    InvalidSidecar(String),
    /// Filesystem I/O failure while reading a sidecar file.
    Io(String),
}

impl fmt::Display for GateSightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid gatesight config: {msg}"),
            Self::DimensionMismatch {
                context,
                expected,
                actual,
            } => write!(
                f,
                "gatesight dimension mismatch in {context}: expected {expected}, got {actual}"
            ),
            Self::InvalidSidecar(msg) => write!(f, "invalid gatesight sidecar: {msg}"),
            Self::Io(msg) => write!(f, "gatesight I/O error: {msg}"),
        }
    }
}

impl std::error::Error for GateSightError {}

/// Convenience result alias for GateSight operations.
pub type Result<T> = std::result::Result<T, GateSightError>;

/// Operating mode for GateSight KV eviction (default [`Self::Disabled`] = bit-identical baseline).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GateSightMode {
    /// Disabled: zero capture overhead and zero KV eviction (bit-identical baseline).
    #[default]
    Disabled,
    /// Path A: zero-training slow-head `block_size = 16` eviction (zero V^T requantization).
    PathABlock16,
    /// Path B: lightweight `867 -> 64 -> 1` per-head MLP token-level eviction.
    PathBMlp,
}

impl GateSightMode {
    /// True when GateSight capture and eviction are active.
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

/// Top-level runtime configuration for GateSight KV eviction.
#[derive(Clone, Debug, PartialEq)]
pub struct GateSightConfig {
    /// Eviction path (defaults to [`GateSightMode::Disabled`]).
    pub mode: GateSightMode,
    /// Target Zone-2 retention fraction `p in (0, 1]`.
    pub target_retention: f32,
    /// Optional path to the authenticated Path-B MLP sidecar file.
    pub sidecar_path: Option<PathBuf>,
    /// Number of preceding GDN layers captured in the L2-resident rolling ring.
    pub window_gdn_layers: usize,
    /// Chunk length (tokens) covered by one rolling-ring pass.
    pub ring_chunk_tokens: usize,
    /// Half-life threshold (tokens) used by the load-time static slow-head classifier.
    pub slow_head_half_life_tokens: f32,
    /// Number of leading attention-sink blocks (`block_size = 16`) always protected in Zone 1.
    pub sink_blocks: usize,
    /// Number of trailing local-window tokens always protected at the end of Zone 2.
    pub recent_window_tokens: usize,
    /// Decay-weighting rule used by Path-A zero-training scoring.
    pub decay_weighting: DecayWeighting,
}

impl Default for GateSightConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl GateSightConfig {
    /// Construct a disabled default configuration (bit-identical baseline).
    #[must_use]
    pub fn new() -> Self {
        Self {
            mode: GateSightMode::Disabled,
            target_retention: 0.40,
            sidecar_path: None,
            window_gdn_layers: DEFAULT_GDN_WINDOW_LAYERS,
            ring_chunk_tokens: DEFAULT_RING_CHUNK_TOKENS,
            slow_head_half_life_tokens: DEFAULT_SLOW_HEAD_HALF_LIFE_TOKENS,
            sink_blocks: 1,
            recent_window_tokens: 256,
            decay_weighting: DecayWeighting::PureWriteNorm,
        }
    }

    /// Parse opt-in configuration from process environment variables.
    ///
    /// Defaults cleanly to [`GateSightMode::Disabled`] when [`GATESIGHT_EVICT_ENV`] is unset.
    /// Fails closed on non-UTF-8 environment values (`VarError::NotUnicode`).
    ///
    /// # Errors
    /// Returns [`GateSightError::InvalidConfig`] on non-UTF-8 or unrecognized mode values,
    /// out-of-range retention ratios, or missing sidecar path when `path_b` is selected.
    pub fn from_env() -> Result<Self> {
        let mode_raw = read_optional_env(GATESIGHT_EVICT_ENV)?;
        let retention_raw = read_optional_env(GATESIGHT_RETENTION_ENV)?;
        let sidecar_raw = read_optional_env(GATESIGHT_SIDECAR_PATH_ENV)?;
        Self::from_env_values(
            mode_raw.as_deref(),
            retention_raw.as_deref(),
            sidecar_raw.as_deref(),
        )
    }

    /// Parse configuration from explicit optional string values (deterministic, testable).
    ///
    /// # Errors
    /// Returns [`GateSightError::InvalidConfig`] when any supplied setting is malformed or
    /// violates cross-field invariants.
    pub fn from_env_values(
        mode_raw: Option<&str>,
        retention_raw: Option<&str>,
        sidecar_raw: Option<&str>,
    ) -> Result<Self> {
        let mut cfg = Self::new();
        cfg.mode = match mode_raw.map(str::trim).filter(|s| !s.is_empty()) {
            None => GateSightMode::Disabled,
            Some(v) => match v.to_ascii_lowercase().as_str() {
                "0" | "off" | "false" | "disabled" => GateSightMode::Disabled,
                "1" | "path_a" | "block16" | "heuristic" => GateSightMode::PathABlock16,
                "2" | "path_b" | "mlp" | "full" => GateSightMode::PathBMlp,
                other => {
                    return Err(GateSightError::InvalidConfig(format!(
                        "{GATESIGHT_EVICT_ENV}={other:?}: expected off|path_a|path_b"
                    )));
                }
            },
        };

        if let Some(raw) = retention_raw.map(str::trim).filter(|s| !s.is_empty()) {
            let p: f32 = raw.parse().map_err(|_| {
                GateSightError::InvalidConfig(format!(
                    "{GATESIGHT_RETENTION_ENV}={raw:?}: expected float in (0.0, 1.0]"
                ))
            })?;
            if !p.is_finite() || p <= 0.0 || p > 1.0 {
                return Err(GateSightError::InvalidConfig(format!(
                    "{GATESIGHT_RETENTION_ENV}={p}: must be finite and in (0.0, 1.0]"
                )));
            }
            cfg.target_retention = p;
        } else if cfg.mode == GateSightMode::PathBMlp {
            cfg.target_retention = 0.25;
        }

        if let Some(raw) = sidecar_raw.map(str::trim).filter(|s| !s.is_empty()) {
            cfg.sidecar_path = Some(PathBuf::from(raw));
        }

        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate structural invariants of this configuration.
    ///
    /// # Errors
    /// Returns [`GateSightError::InvalidConfig`] if any field is out of range.
    pub fn validate(&self) -> Result<()> {
        if !self.target_retention.is_finite()
            || self.target_retention <= 0.0
            || self.target_retention > 1.0
        {
            return Err(GateSightError::InvalidConfig(format!(
                "target_retention={} must be in (0.0, 1.0]",
                self.target_retention
            )));
        }
        if self.window_gdn_layers == 0 {
            return Err(GateSightError::InvalidConfig(
                "window_gdn_layers must be > 0".to_string(),
            ));
        }
        if self.ring_chunk_tokens == 0 {
            return Err(GateSightError::InvalidConfig(
                "ring_chunk_tokens must be > 0".to_string(),
            ));
        }
        if !self.slow_head_half_life_tokens.is_finite() || self.slow_head_half_life_tokens <= 0.0 {
            return Err(GateSightError::InvalidConfig(format!(
                "slow_head_half_life_tokens={} must be > 0",
                self.slow_head_half_life_tokens
            )));
        }
        if self.mode == GateSightMode::PathBMlp && self.sidecar_path.is_none() {
            return Err(GateSightError::InvalidConfig(format!(
                "path_b requires {GATESIGHT_SIDECAR_PATH_ENV} to point to an ATLGTS01 sidecar"
            )));
        }
        Ok(())
    }
}

fn read_optional_env(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(GateSightError::InvalidConfig(format!(
            "{name} contains non-UTF-8 bytes"
        ))),
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GateSightConfig>();
    assert_send_sync::<GateSightError>();
    assert_send_sync::<ModelSlowHeadTable>();
    assert_send_sync::<GdnSignalRingDescriptor>();
    assert_send_sync::<GateSightMlpSidecar>();
};
