// SPDX-License-Identifier: AGPL-3.0-only

//! `recipe.toml` schema, version 1. Unknown keys are rejected everywhere so a
//! typo cannot silently fall back to a default.

use serde::Deserialize;
use std::collections::BTreeMap;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeFile {
    pub schema: u32,
    pub id: String,
    #[serde(default)]
    pub description: String,
    pub checkpoint: Checkpoint,
    pub precision: PrecisionPlan,
    /// NUMERICS gates outside the precision plan, pinned to exact values.
    #[serde(default)]
    pub numerics_env: BTreeMap<String, String>,
    /// ROUTE gates (claimed output-identical) pinned to exact values.
    #[serde(default)]
    pub route_env: BTreeMap<String, String>,
    #[serde(default)]
    pub route_policy: RoutePolicy,
    pub sampling: Sampling,
    pub limits: Limits,
    pub kv_cache_dtype: String,
    pub speculation: Speculation,
    /// Numerics-relevant serve flags (see `facts::CLI_NUMERICS`) pinned to a
    /// value. An undeclared one must stay at its built-in default.
    #[serde(default)]
    pub cli: BTreeMap<String, String>,
    /// RESOURCE/OBSERVABILITY/DEBUG vars with a declared allowed range.
    #[serde(default)]
    pub operational_env: BTreeMap<String, OperationalRule>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    /// `config.json` `model_type`, checked against the loaded checkpoint.
    pub model_type: String,
    pub model_id: String,
    pub revision: String,
    /// File name (relative to the model directory) to lowercase hex sha256.
    pub sha256: BTreeMap<String, String>,
}

/// Served format per module family. The selectors live in `precision.rs`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecisionPlan {
    pub router: DenseFormat,
    pub hyper_connection: DenseFormat,
    pub gdn_proj: DenseFormat,
    pub attention_proj: DenseFormat,
    pub shared_expert: DenseFormat,
    pub lm_head: DenseFormat,
    pub prefill_dense: PrefillFormat,
    pub routed_experts: RoutedFormat,
    pub gdn_state: StateFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseFormat {
    /// Checkpoint BF16/MXFP8 re-quantized to NVFP4 at load (today's default).
    Nvfp4Requant,
    /// Served from the checkpoint at BF16.
    Bf16,
    /// Per-row E4M3 (the `DENSE_FP8` production path).
    Fp8E4m3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillFormat {
    /// The FAST_PROJ / FP8 prefill projections the env pins describe.
    FastFp8,
    /// `ATLAS_QWEN4_PREFILL_NATIVE`: prefill uses the decode plan's formats.
    Native,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutedFormat {
    Nvfp4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateFormat {
    Fp32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutePolicy {
    /// Undeclared ROUTE vars are allowed but listed in the receipt.
    #[default]
    Flag,
    /// Undeclared ROUTE vars refuse startup.
    Refuse,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sampling {
    pub defaults: SamplingValues,
    #[serde(default)]
    pub thinking: Option<SamplingValues>,
    pub ranges: SamplingRanges,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingValues {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u64,
    pub presence_penalty: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingRanges {
    pub temperature: [f64; 2],
    pub top_p: [f64; 2],
    pub top_k: [u64; 2],
    pub presence_penalty: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    /// Inclusive `[min, max]` for `--max-seq-len`.
    pub max_seq_len: [u64; 2],
    pub max_prefill_tokens: [u64; 2],
    pub max_batch_size: [u64; 2],
    pub qwen4_qsa: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Speculation {
    pub mode: SpeculationMode,
    #[serde(default)]
    pub num_drafts: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeculationMode {
    /// Target-only: no speculative method may be enabled.
    Off,
    /// Native MTP (`--speculative`) with an exact `num_drafts`.
    Mtp,
}

/// Exactly one of `values` or `min`/`max` (integers, inclusive).
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationalRule {
    #[serde(default)]
    pub values: Option<Vec<String>>,
    #[serde(default)]
    pub min: Option<i64>,
    #[serde(default)]
    pub max: Option<i64>,
}

impl OperationalRule {
    pub fn allows(&self, raw: &str) -> bool {
        if let Some(values) = &self.values {
            return values.iter().any(|v| v == raw);
        }
        let Ok(n) = raw.trim().parse::<i64>() else {
            return false;
        };
        self.min.is_none_or(|lo| n >= lo) && self.max.is_none_or(|hi| n <= hi)
    }
}
