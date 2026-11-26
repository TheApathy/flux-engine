// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash drafter weight loader.
//!
//! Loads `z-lab/Qwen3.6-{27B,35B-A3B}-DFlash`-style drafter checkpoints into
//! the typed [`DflashWeights`] structure consumed by
//! [`crate::layers::BlockDiffusionDraftHead`]. The drafter is a small
//! Qwen3-architecture transformer (8 layers, hidden=2048, GQA 32:4) with
//! these distinctive parts vs. a vanilla Qwen3:
//!
//!  * `fc.weight` (or `model.fc.weight`) —
//!    `[draft_hidden, len(target_layer_ids) * target_hidden]` BF16 projection
//!    that maps the stack of captured target hidden states into the drafter's
//!    input space.
//!  * `hidden_norm.weight` (or `model.hidden_norm.weight`) — RMSNorm applied
//!    to the projected target context before mixing with token embeddings.
//!  * `embed_tokens` and `lm_head` — absent from the drafter checkpoint; the
//!    runtime shares both from the target model at construction time.
//!  * Optional `d2t` — draft-vocab → target-vocab id remap (absent when
//!    drafter shares vocab with target, as in Qwen3.6-35B-A3B-DFlash where
//!    both = 248320).
//!  * Special `mask_token_id` (`248070` for Qwen3.6-DFlash) used for the γ
//!    "to-be-predicted" positions in block diffusion.
//!
//! Under TP the drafter is **not sharded** — it's small (~1–2 GB BF16),
//! every rank loads the full set. Mirrors the existing MTP-under-TP pattern
//! (`MTP loads ALL experts on every rank — no EP all_reduce needed`).

use anyhow::{Context, Result};
use serde::Deserialize;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::dflash_validation::validate_dflash_store;
use crate::weight_map::{DenseWeight, QuantizedWeight, dense, quantized_modelopt_with_shape};

/// ModelOpt metadata required to admit pre-quantized DFlash2 projections.
/// Other quantizers remain unsupported and fail during strict tensor admission.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashQuantizationConfig {
    pub quant_method: String,
    pub quant_algo: String,
    pub group_size: usize,
    #[serde(default)]
    pub producer: Option<DflashQuantizationProducer>,
}

/// Exact producer receipt for calibrated DFlash2 activation scales.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashQuantizationProducer {
    pub name: String,
    pub version: String,
}

impl DflashQuantizationConfig {
    /// The only checkpoint schema whose activation scales Atlas may consume.
    pub(crate) fn is_exact_dflash2_rtncal(&self) -> bool {
        self.quant_method == "modelopt"
            && self.quant_algo == "NVFP4"
            && self.group_size == 16
            && self.producer.as_ref().is_some_and(|producer| {
                producer.name == "modelopt"
                    && producer.version == "dflash2-nvfp4-rtn-calibrated-1.0"
            })
    }
}

/// Checkpoint-family contract for the otherwise shared DFlash/DSpark loader.
///
/// The family distinction is semantic, not cosmetic (Markov/confidence heads
/// belong only to DSpark). Legacy DFlash counts the known anchor in
/// `block_size`; published DSpark checkpoints count only the draft rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterCheckpointFamily {
    Dflash,
    Dspark,
}

/// Width contract for a checkpoint-backed parallel draft pass.
///
/// Current vLLM DSpark treats `num_speculative_tokens=2*block_size` as one
/// wider query block, not two feedback passes.  Keep that experimental parity
/// escape hatches explicit and bounded: DSpark accepts only its exact 2x case,
/// while an attested Qwen3.8 DFlash2 B8 checkpoint may execute B16/gamma15.
fn draft_width_is_supported(
    family: DrafterCheckpointFamily,
    trained_drafts: usize,
    requested_drafts: usize,
    allow_dspark_2x: bool,
    allow_dflash2_b16: bool,
) -> bool {
    requested_drafts <= trained_drafts
        || (family == DrafterCheckpointFamily::Dspark
            && allow_dspark_2x
            && requested_drafts == trained_drafts.saturating_mul(2))
        || (family == DrafterCheckpointFamily::Dflash
            && allow_dflash2_b16
            && trained_drafts == 7
            && requested_drafts == 15)
}

/// DSpark target-verify planner mode. SGLang PR 34966 defaults to `static`,
/// which verifies the full gamma+1 window and does not consume confidence.
/// The dynamic modes need ragged layouts and a hardware-profiled SPS table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsparkVerifyMode {
    Static,
    CapAccept,
    Compact,
}

impl std::fmt::Display for DsparkVerifyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static => f.write_str("static"),
            Self::CapAccept => f.write_str("cap-accept"),
            Self::Compact => f.write_str("compact"),
        }
    }
}

impl std::str::FromStr for DsparkVerifyMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "static" => Ok(Self::Static),
            "cap-accept" => Ok(Self::CapAccept),
            "compact" => Ok(Self::Compact),
            _ => anyhow::bail!(
                "unsupported DSpark verify mode {value:?}; expected static, cap-accept, or compact"
            ),
        }
    }
}

/// Resolved learned-confidence feature contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsparkConfidenceConfig {
    pub with_markov: bool,
    pub input_dim: usize,
}

/// DSpine (`arXiv:2609.36173`) `dflash_config.selected_prefix_code_bridge` block.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DspineBridgeConfig {
    pub code_size: usize,
    pub rank: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub eps: f32,
    #[serde(default)]
    pub kappa: Option<f32>,
}

/// DSpine `dflash_config.selected_prefix_last_write_refine` block.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DspineRefineConfig {
    pub kind: String,
    #[serde(default = "default_dspine_refine_version")]
    pub version: u32,
    pub head_topk: usize,
    #[serde(default = "default_dspine_refine_weight")]
    pub weight: f32,
}

/// DSpine root `code_bridge_second_pass` block.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DspineSecondPass {
    pub mode: String,
    pub first_exact: usize,
    pub head_topk: usize,
}

/// Validated DSpine code-bridge execution specification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DspineSpec {
    pub code_size: usize,
    pub rank: usize,
    pub eps: f32,
    pub head_topk: usize,
    /// Bitmask of 0-based active bridge layers (`(mask >> layer_idx) & 1 != 0`).
    pub active_layers_mask: u64,
    pub terminal_residual_rank: usize,
}

impl DspineSpec {
    #[inline]
    pub fn is_layer_active(self, layer_idx: usize) -> bool {
        layer_idx < 64 && ((self.active_layers_mask >> layer_idx) & 1) != 0
    }
}

fn default_dspine_refine_version() -> u32 {
    1
}

fn default_dspine_refine_weight() -> f32 {
    1.0
}

const ROOT_DSPINE_SECOND_PASS_KEY: &str = "_root_code_bridge_second_pass";

impl std::fmt::Display for DrafterCheckpointFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dflash => f.write_str("dflash"),
            Self::Dspark => f.write_str("dspark"),
        }
    }
}

/// Drafter HF `config.json` (subset Atlas consumes). Mirrors
/// `z-lab/Qwen3.6-35B-A3B-DFlash/config.json` field names verbatim so
/// `serde_json::from_str` works directly on the raw file.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashConfig {
    /// HF model class. `DSparkDraftModel` is load-bearing: it selects the
    /// DSpark Markov/confidence feature contract.
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Underlying Transformers decoder ABI. Native Flash-Next V3 requires
    /// the Qwen3 drafter implementation explicitly rather than accepting an
    /// unknown architecture that happens to share tensor names.
    #[serde(default)]
    pub model_type: Option<String>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    /// Decoder depth of the target used to train this drafter. Newer
    /// Qwen3.8 DFlash checkpoints serialize this explicitly.
    #[serde(default)]
    pub num_target_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub draft_vocab_size: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Family-specific training width. DFlash uses `B = anchor + drafts`;
    /// DSpark uses `B = drafts` and adds the target bonus only at verification.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// `block_size` has a legacy default, but native Flash-Next must attest
    /// that the root field was actually serialized. Populated only by
    /// [`parse_dflash_config`]; direct serde callers remain un-attested.
    #[serde(skip)]
    pub(crate) root_block_size_explicit: bool,
    /// DFlash-specific nested config object.
    #[serde(default)]
    pub dflash_config: Option<DflashSubConfig>,
    /// Optional top-level `mask_token_id` serialized by flat-root checkpoints
    /// (e.g. `Qwen3DSparkModel`), normalized into `dflash_config` by
    /// [`parse_dflash_config`].
    #[serde(default)]
    pub mask_token_id: Option<u32>,
    /// Optional top-level `target_layer_ids` serialized by flat-root
    /// checkpoints (e.g. `Qwen3DSparkModel`), normalized into `dflash_config`
    /// by [`parse_dflash_config`].
    #[serde(default)]
    pub target_layer_ids: Option<Vec<usize>>,
    /// Optional top-level target model type hint (e.g. `"qwen4_exp"`).
    #[serde(default)]
    pub target_model_type: Option<String>,
    /// Per-layer attention type. Qwen3.6-27B-DFlash ships
    /// `["sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"]`
    /// — first 4 layers are SWA, last is full. Older drafters omit the field
    /// (treated as all `full_attention`).
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// SWA span (only meaningful when `layer_types` contains `sliding_attention`).
    /// Qwen3.6-27B-DFlash sets this to 2048.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Whether the drafter's block attention is causal.
    ///
    /// DFlash2 checkpoints ship `"is_causal": false` — the draft block is
    /// attended BIDIRECTIONALLY, because every masked position is predicted
    /// simultaneously and each may see all the others. Until 2026-08-20 this
    /// field was not parsed at all, so `from_weights.rs` derived causality
    /// purely from `layer_types` (`causals.push(is_sliding)`). DFlash2's
    /// `layer_types` is five `sliding_attention` entries and no
    /// `full_attention`, so every layer was run CAUSAL — the opposite of how
    /// the checkpoint was trained. Measured effect: acceptance 2.75/7 (39%)
    /// against 5.05/7 (72%) for the causally-trained qwen38-v2 drafter, i.e.
    /// dflash2 read as a large regression (28.28 vs 41.22 tok/s) when the
    /// mismatch alone could account for it.
    ///
    /// `None` (field absent) keeps the historical `layer_types`-derived
    /// behaviour, so every existing drafter is bit-identical.
    #[serde(default)]
    pub is_causal: Option<bool>,
    /// RMSNorm epsilon used by every drafter norm. Atlas's DFlash kernels use
    /// the Qwen default, but strict checkpoint admission must still reject a
    /// config trained with a different value.
    #[serde(default)]
    pub rms_norm_eps: Option<f32>,
    /// FFN activation serialized by Qwen configs. Existing drafters omit this
    /// field or use SiLU, which is the activation Atlas executes.
    #[serde(default)]
    pub hidden_act: Option<String>,
    /// RoPE rotation base. Qwen3.6-27B-DFlash and 35B-DFlash both ship 10_000_000.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// RoPE scaling block. Older Qwen configs call this `rope_scaling`, while
    /// Transformers 5 / Qwen3.8 serializes the same block as
    /// `rope_parameters`. Accept both spellings: silently treating a Qwen3.8
    /// DSpark checkpoint as vanilla RoPE destroys draft acceptance.
    #[serde(default, alias = "rope_parameters")]
    pub rope_scaling: Option<DflashRopeScaling>,
    /// DSpark low-rank Markov head rank `r` (config key `markov_rank`).
    /// `0` (or absent) ⇒ no Markov head. The DSpark-AEON-draft checkpoint
    /// ships `markov_rank: 256` with a rank-256 `VanillaMarkov` head that
    /// adds a per-position bigram logit bias `B(prev) = W2(W1[prev])` and
    /// samples the block LEFT-TO-RIGHT (each position's chosen token biases
    /// the next), matching the upstream RadixArk/SpecForge DSpark head.
    #[serde(default)]
    pub markov_rank: usize,
    /// DSpark Markov head variant (config key `markov_head_type`). Only
    /// `"vanilla"` is supported; any other value (including `"gated"`) is
    /// rejected fail-closed during config and store validation.
    #[serde(default = "default_markov_head_type")]
    pub markov_head_type: String,
    /// Learned DSpark acceptance-confidence head. Atlas must not silently
    /// ignore this: the head changes dynamic/ragged verify scheduling.
    #[serde(default)]
    pub enable_confidence_head: Option<bool>,
    #[serde(default)]
    pub confidence_head_with_markov: Option<bool>,
    /// Present only when the drafter projections were exported pre-quantized.
    /// Atlas currently accepts exactly ModelOpt NVFP4 with 16-value groups.
    #[serde(default)]
    pub quantization_config: Option<DflashQuantizationConfig>,
}

impl DflashConfig {
    /// Effective RMSNorm epsilon for legacy configs that predate this parsed
    /// field. Strict official profiles inspect the raw [`Option`] so omission
    /// remains distinguishable from an explicit declaration.
    pub fn resolved_rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps.unwrap_or_else(default_rms_norm_eps)
    }

    /// Effective activation for legacy configs that omit `hidden_act`.
    /// Strict official profiles require the raw field to be present.
    pub fn resolved_hidden_act(&self) -> &str {
        self.hidden_act.as_deref().unwrap_or("silu")
    }

    /// Resolve the training block width. Generic DFlash drafters (z-lab)
    /// serialize `block_size` at the config root; DFlash 2 (incoai) nests it
    /// inside `dflash_config` (block = 1 anchor + drafts). The nested value
    /// wins when present.
    pub fn resolved_block_size(&self) -> usize {
        self.dflash_config
            .as_ref()
            .and_then(|sub| (sub.block_size.is_some()).then(|| sub.block_size.expect("checked")))
            .unwrap_or(self.block_size)
    }

    /// True when the checkpoint declares the DFlash 2 contract: a
    /// `dflash_config.conv_kernel_size > 1` plus a candidate selector width.
    /// Both the conv and the selector must be executed together — the released
    /// checkpoint trains them jointly and neither can be dropped silently.
    pub fn is_dflash2(&self) -> bool {
        self.dflash_config.as_ref().is_some_and(|sub| {
            sub.conv_kernel_size > 1 && sub.selector_rank > 0 && sub.selector_top_k > 0
        })
    }

    /// Resolve the checkpoint family from explicit HF declarations.
    /// A disagreement fails closed instead of selecting block semantics from
    /// incidental tensors such as the Markov weights.
    pub fn checkpoint_family(&self) -> Result<DrafterCheckpointFamily> {
        let architecture_is_qwen3_dspark = self
            .architectures
            .iter()
            .any(|name| name == "Qwen3DSparkModel");
        let architecture_is_dspark = self
            .architectures
            .iter()
            .any(|name| name == "DSparkDraftModel");
        let projector = self
            .dflash_config
            .as_ref()
            .and_then(|sub| sub.projector_type.as_deref());
        let projector_is_dspark = projector == Some("dspark");

        if architecture_is_qwen3_dspark {
            anyhow::ensure!(
                projector.is_none() || projector_is_dspark,
                "Drafter architecture is Qwen3DSparkModel but \
                 dflash_config.projector_type={projector:?}; expected None or `dspark`"
            );
            return Ok(DrafterCheckpointFamily::Dspark);
        }

        match (architecture_is_dspark, projector_is_dspark) {
            (true, true) => Ok(DrafterCheckpointFamily::Dspark),
            (false, false) => Ok(DrafterCheckpointFamily::Dflash),
            (true, false) => anyhow::bail!(
                "Drafter architecture is DSparkDraftModel but \
                 dflash_config.projector_type={projector:?}; expected `dspark`"
            ),
            (false, true) => anyhow::bail!(
                "Drafter declares dflash_config.projector_type=`dspark` but architectures \
                 does not contain `DSparkDraftModel`"
            ),
        }
    }

    fn resolve_confidence_bool(
        &self,
        name: &str,
        top: Option<bool>,
        nested: Option<bool>,
        default: bool,
    ) -> Result<bool> {
        if let (Some(top), Some(nested)) = (top, nested) {
            anyhow::ensure!(
                top == nested,
                "DSpark confidence config disagrees for `{name}`: top-level={top}, \
                 dflash_config={nested}"
            );
        }
        Ok(top.or(nested).unwrap_or(default))
    }

    /// Resolve the trained confidence feature shape. This validates config
    /// semantics only; runtime planner support is checked separately.
    pub fn confidence_head_config(&self) -> Result<Option<DsparkConfidenceConfig>> {
        let family = self.checkpoint_family()?;
        let nested = self.dflash_config.as_ref();
        let enabled = self.resolve_confidence_bool(
            "enable_confidence_head",
            self.enable_confidence_head,
            nested.and_then(|sub| sub.enable_confidence_head),
            false,
        )?;
        let with_markov = self.resolve_confidence_bool(
            "confidence_head_with_markov",
            self.confidence_head_with_markov,
            nested.and_then(|sub| sub.confidence_head_with_markov),
            self.markov_rank > 0,
        )?;
        if !enabled {
            anyhow::ensure!(
                !with_markov
                    || (self.confidence_head_with_markov.is_none()
                        && nested
                            .and_then(|sub| sub.confidence_head_with_markov)
                            .is_none()),
                "confidence_head_with_markov=true requires enable_confidence_head=true"
            );
            return Ok(None);
        }
        anyhow::ensure!(
            family == DrafterCheckpointFamily::Dspark,
            "learned confidence-head semantics are supported only for explicit DSpark checkpoints"
        );
        if with_markov {
            anyhow::ensure!(
                self.markov_rank > 0,
                "confidence_head_with_markov=true requires markov_rank > 0"
            );
        }
        let input_dim = self
            .hidden_size
            .checked_add(if with_markov { self.markov_rank } else { 0 })
            .ok_or_else(|| anyhow::anyhow!("DSpark confidence input width overflow"))?;
        Ok(Some(DsparkConfidenceConfig {
            with_markov,
            input_dim,
        }))
    }

    /// Validate features whose silent omission would change checkpoint
    /// semantics. Kept separate from tensor validation so the server rejects
    /// an unsupported config before loading its multi-GB weight file.
    pub fn validate_supported_semantics(&self) -> Result<DrafterCheckpointFamily> {
        let family = self.checkpoint_family()?;
        if self.markov_rank > 0 {
            anyhow::ensure!(
                self.markov_head_type.eq_ignore_ascii_case("vanilla"),
                "Flux DSpark/Markov supports only markov_head_type=`vanilla`; got {:?}",
                self.markov_head_type
            );
        }
        self.confidence_head_config()?;
        self.dspine_spec()?;
        Ok(family)
    }

    /// True when the checkpoint declares causal draft block attention via root
    /// `is_causal: true` or `dflash_config.draft_block_causal: true`.
    pub fn is_draft_block_causal(&self) -> bool {
        let nested_causal = self
            .dflash_config
            .as_ref()
            .and_then(|sub| sub.unknown_fields.get("draft_block_causal"))
            .and_then(serde_json::Value::as_bool);
        if self.is_causal == Some(false) || nested_causal == Some(false) {
            return false;
        }
        self.is_causal == Some(true) || nested_causal == Some(true)
    }

    /// True when this checkpoint declares a valid DSpine code bridge.
    pub fn is_dspine(&self) -> bool {
        self.dspine_spec().ok().flatten().is_some()
    }

    /// Validate and resolve the DSpine (`selected_prefix_code_bridge`) contract
    /// when declared, failing closed on unsupported versions, non-independent
    /// write modes, active terminal heads, or refinement mismatches.
    pub fn dspine_spec(&self) -> Result<Option<DspineSpec>> {
        let Some(sub) = self.dflash_config.as_ref() else {
            return Ok(None);
        };
        let get_opt = |key: &str| -> Option<&serde_json::Value> {
            sub.unknown_fields.get(key).filter(|v| !v.is_null())
        };

        let Some(bridge_val) = get_opt("selected_prefix_code_bridge") else {
            anyhow::ensure!(
                get_opt(ROOT_DSPINE_SECOND_PASS_KEY).is_none()
                    && get_opt("selected_prefix_last_write_refine").is_none(),
                "DSpine second-pass / last-write refinement fields require \
                 `dflash_config.selected_prefix_code_bridge`"
            );
            if let Some(tr_val) = get_opt("selected_prefix_terminal_residual_rank") {
                let tr: usize = serde_json::from_value(tr_val.clone())
                    .context("Parsing dflash_config.selected_prefix_terminal_residual_rank")?;
                anyhow::ensure!(
                    tr == 0,
                    "dflash_config.selected_prefix_terminal_residual_rank={tr} requires \
                     `selected_prefix_code_bridge`"
                );
            }
            return Ok(None);
        };

        let bridge: DspineBridgeConfig = serde_json::from_value(bridge_val.clone())
            .context("Parsing dflash_config.selected_prefix_code_bridge")?;
        anyhow::ensure!(
            bridge.code_size > 0
                && (1..=512).contains(&bridge.rank)
                && bridge.eps > 0.0
                && bridge.eps.is_finite(),
            "Unsupported DSpine code-bridge geometry: code_size={}, rank={} (supported 1..=512), eps={}",
            bridge.code_size,
            bridge.rank,
            bridge.eps,
        );
        anyhow::ensure!(
            (1..=64).contains(&self.num_hidden_layers),
            "DSpine num_hidden_layers={} must be in 1..=64",
            self.num_hidden_layers
        );
        anyhow::ensure!(
            sub.conv_kernel_size == 0
                && sub.conv_group_size == 0
                && sub.selector_rank == 0
                && sub.selector_top_k == 0
                && self.markov_rank == 0
                && sub.projector_type.is_none(),
            "DSpine checkpoints cannot combine with DFlash2 conv/selector or DSpark Markov heads"
        );

        let parse_bool = |key: &str| -> Result<Option<bool>> {
            get_opt(key)
                .map(|v| {
                    v.as_bool()
                        .ok_or_else(|| anyhow::anyhow!("dflash_config.{key} must be a boolean"))
                })
                .transpose()
        };
        if let Some(terminal) = parse_bool("selected_prefix_terminal")? {
            anyhow::ensure!(
                !terminal,
                "DSpine requires selected_prefix_terminal=false (standalone terminal head is disabled)"
            );
        }
        if let Some(writes) = parse_bool("selected_prefix_feature_writes")? {
            anyhow::ensure!(
                writes,
                "DSpine requires selected_prefix_feature_writes=true"
            );
        }
        if let Some(mode_val) = get_opt("selected_prefix_feature_write_mode") {
            let mode = mode_val.as_str().ok_or_else(|| {
                anyhow::anyhow!("dflash_config.selected_prefix_feature_write_mode must be a string")
            })?;
            anyhow::ensure!(
                mode == "independent",
                "DSpine supports only selected_prefix_feature_write_mode=\"independent\", got {mode:?}"
            );
        }
        anyhow::ensure!(
            get_opt("selected_prefix_feature_reliability").is_none(),
            "DSpine does not support selected_prefix_feature_reliability"
        );
        if let Some(next_rows) = parse_bool("selected_prefix_feature_next_token_rows")? {
            anyhow::ensure!(
                !next_rows,
                "DSpine requires selected_prefix_feature_next_token_rows=false"
            );
        }

        let nested_causal = parse_bool("draft_block_causal")?;
        anyhow::ensure!(
            self.is_causal != Some(false) && nested_causal != Some(false),
            "DSpine requires causal draft block attention (is_causal / draft_block_causal cannot be false)"
        );
        anyhow::ensure!(
            self.is_causal == Some(true) || nested_causal == Some(true),
            "DSpine requires explicit causal draft block attention (`is_causal: true` or \
             `dflash_config.draft_block_causal: true`)"
        );

        let active_layers_mask = if let Some(layers_val) = get_opt("selected_prefix_feature_layers")
        {
            let layers_1based: Vec<usize> = serde_json::from_value(layers_val.clone())
                .context("Parsing dflash_config.selected_prefix_feature_layers")?;
            anyhow::ensure!(
                !layers_1based.is_empty(),
                "DSpine selected_prefix_feature_layers must not be empty"
            );
            let mut mask = 0u64;
            for l1 in layers_1based {
                anyhow::ensure!(
                    (1..=self.num_hidden_layers).contains(&l1),
                    "DSpine selected_prefix_feature_layers entry {l1} out of 1..={}",
                    self.num_hidden_layers
                );
                let bit = 1u64 << (l1 - 1);
                anyhow::ensure!(
                    (mask & bit) == 0,
                    "DSpine selected_prefix_feature_layers contains duplicate layer {l1}"
                );
                mask |= bit;
            }
            let last_bit = 1u64 << (self.num_hidden_layers - 1);
            anyhow::ensure!(
                (mask & last_bit) != 0,
                "DSpine requires the final draft layer ({}) to be active in selected_prefix_feature_layers",
                self.num_hidden_layers
            );
            mask
        } else if self.num_hidden_layers == 64 {
            u64::MAX
        } else {
            (1u64 << self.num_hidden_layers) - 1
        };

        let second_pass: Option<DspineSecondPass> = get_opt(ROOT_DSPINE_SECOND_PASS_KEY)
            .map(|v| serde_json::from_value(v.clone()).context("Parsing code_bridge_second_pass"))
            .transpose()?;
        if let Some(sp) = &second_pass {
            anyhow::ensure!(
                sp.mode == "serial_table",
                "DSpine supports only code_bridge_second_pass.mode=\"serial_table\", got {:?}",
                sp.mode
            );
            anyhow::ensure!(
                sp.first_exact == self.num_hidden_layers - 1,
                "DSpine serial_table redoes the final write only: expected first_exact={}, got {}",
                self.num_hidden_layers - 1,
                sp.first_exact
            );
        }

        let refine: Option<DspineRefineConfig> = get_opt("selected_prefix_last_write_refine")
            .map(|v| {
                serde_json::from_value(v.clone())
                    .context("Parsing dflash_config.selected_prefix_last_write_refine")
            })
            .transpose()?;
        if let Some(rf) = &refine {
            anyhow::ensure!(
                rf.kind == "exact_sender"
                    && rf.version == 1
                    && rf.weight >= 0.0
                    && rf.weight.is_finite(),
                "Unsupported DSpine selected_prefix_last_write_refine: kind={:?}, version={}, weight={}",
                rf.kind,
                rf.version,
                rf.weight,
            );
        }

        let head_topk = match (
            second_pass.as_ref().map(|s| s.head_topk),
            refine.as_ref().map(|r| r.head_topk),
        ) {
            (Some(a), Some(b)) => {
                anyhow::ensure!(
                    a == b,
                    "DSpine head_topk disagrees between code_bridge_second_pass ({a}) and \
                     selected_prefix_last_write_refine ({b})"
                );
                a
            }
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => {
                anyhow::bail!(
                    "DSpine config declares no refinement width: provide \
                     `code_bridge_second_pass` or `dflash_config.selected_prefix_last_write_refine`"
                );
            }
        };
        anyhow::ensure!(
            (1..=16).contains(&head_topk) && head_topk.is_power_of_two(),
            "DSpine head_topk={} must be a power of two in 1..=16",
            head_topk
        );

        let terminal_residual_rank =
            if let Some(tr_val) = get_opt("selected_prefix_terminal_residual_rank") {
                serde_json::from_value(tr_val.clone())
                    .context("Parsing dflash_config.selected_prefix_terminal_residual_rank")?
            } else {
                0
            };
        anyhow::ensure!(
            terminal_residual_rank <= bridge.rank,
            "DSpine selected_prefix_terminal_residual_rank ({terminal_residual_rank}) exceeds rank ({})",
            bridge.rank
        );
        if terminal_residual_rank > 0 {
            anyhow::ensure!(
                refine.is_some()
                    && parse_bool("selected_prefix_feature_gt_training")? == Some(true),
                "DSpine terminal token residual requires selected_prefix_last_write_refine and \
                 selected_prefix_feature_gt_training=true"
            );
        }

        Ok(Some(DspineSpec {
            code_size: bridge.code_size,
            rank: bridge.rank,
            eps: bridge.eps,
            head_topk,
            active_layers_mask,
            terminal_residual_rank,
        }))
    }

    /// Keep dynamic confidence scheduling fail-closed until Flux has the
    /// ragged verify layout and profiled SPS cost table used by SGLang.
    pub fn validate_verify_mode(&self, mode: DsparkVerifyMode) -> Result<()> {
        let family = self.validate_supported_semantics()?;
        if family == DrafterCheckpointFamily::Dflash {
            anyhow::ensure!(
                mode == DsparkVerifyMode::Static,
                "DSpark verify mode {mode} cannot be applied to a DFlash checkpoint"
            );
            return Ok(());
        }
        anyhow::ensure!(
            mode == DsparkVerifyMode::Static,
            "DSpark verify mode {mode} is not implemented in Flux: dynamic cap-accept/compact \
             requires ragged verify layouts plus a hardware-profiled SPS cost table; use \
             --dspark-verify-mode static (full gamma+1 verify, matching SGLang PR 34966)"
        );
        Ok(())
    }

    /// Resolve the requested proposal count under this checkpoint family's
    /// training-width contract.
    pub fn resolve_draft_count(&self, requested: Option<usize>) -> Result<usize> {
        let family = self.validate_supported_semantics()?;
        let block_size = self.resolved_block_size();
        let trained_drafts = match family {
            DrafterCheckpointFamily::Dflash => {
                anyhow::ensure!(
                    block_size >= 2,
                    "DFlash checkpoint block_size={} cannot contain an anchor plus a draft",
                    block_size
                );
                block_size - 1
            }
            DrafterCheckpointFamily::Dspark => {
                anyhow::ensure!(
                    block_size >= 1,
                    "DSpark checkpoint block_size must contain at least one draft"
                );
                block_size
            }
        };
        let drafts = requested.unwrap_or(trained_drafts);
        anyhow::ensure!(drafts > 0, "{family} draft count must be non-zero");
        let dspark_width_multiple = family == DrafterCheckpointFamily::Dspark
            && std::env::var("ATLAS_DSPARK_WIDTH_MULTIPLE").ok().as_deref() == Some("1");
        let dflash2_b16 = family == DrafterCheckpointFamily::Dflash
            && std::env::var("ATLAS_DFLASH2_WIDTH_16").ok().as_deref() == Some("1")
            && super::dflash_validation::exact_dflash2_profile(self)
                .is_ok_and(|profile| profile.wide_verify_k == Some(16));
        anyhow::ensure!(
            draft_width_is_supported(
                family,
                trained_drafts,
                drafts,
                dspark_width_multiple,
                dflash2_b16,
            ),
            "{family} requested {drafts} drafts, but checkpoint block_size={} supports at most \
             {trained_drafts} by default. DSpark permits exactly a 2x trained-width single \
             parallel pass only with ATLAS_DSPARK_WIDTH_MULTIPLE=1. Exact Qwen3.8 DFlash2 B8 \
             permits only B16/gamma15 with ATLAS_DFLASH2_WIDTH_16=1; use --dflash-gamma \
             {trained_drafts} or smaller otherwise",
            block_size
        );
        if dflash2_b16 && drafts == 15 {
            tracing::warn!(
                "EXPERIMENTAL Qwen3.8 DFlash2 wide pass admitted explicitly: trained B8, \
                 serving B16/gamma15 with target verify K16"
            );
        }
        if drafts < trained_drafts {
            tracing::info!(
                "{family} draft width capped to {drafts}; checkpoint supports {trained_drafts} \
                 trained drafts (block_size={block_size})",
            );
        }
        Ok(drafts)
    }
}

fn default_markov_head_type() -> String {
    "vanilla".to_string()
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

/// HF `rope_scaling` block. Qwen3.5/3.6 DFlash drafters that DO use scaling
/// set `rope_type = "yarn"`. The 27B drafter omits the entire block.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashRopeScaling {
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    /// Transformers 5 places the RoPE base inside `rope_parameters` instead
    /// of at top level. Older configs leave this absent and use
    /// [`DflashConfig::rope_theta`].
    #[serde(default)]
    pub rope_theta: Option<f32>,
    /// Explicit YaRN cos/sin multiplier. When absent, HF derives
    /// `1 + 0.1 * ln(factor)`.
    #[serde(default)]
    pub attention_factor: Option<f32>,
}

fn default_block_size() -> usize {
    16
}

/// Nested `dflash_config` block in the drafter's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashSubConfig {
    /// DFlash 2 serializes the training block width here (top-level
    /// `block_size` is absent in incoai checkpoints). `None` for z-lab
    /// drafters which use the config-root `block_size`.
    #[serde(default)]
    pub block_size: Option<usize>,
    /// Token id used to fill the γ "to-be-predicted" positions during draft
    /// inference. `248070` for Qwen3.6-DFlash.
    pub mask_token_id: u32,
    /// Target-model layer indices to capture intermediate hidden states from.
    /// `[1, 10, 19, 28, 37]` for Qwen3.6-35B-A3B-DFlash. Order matters:
    /// shallow-to-deep concatenation is what `fc` expects.
    pub target_layer_ids: Vec<usize>,
    /// Published SpecForge DSpark checkpoints use `projector_type="dspark"`.
    /// It must agree with the HF `architectures` declaration.
    #[serde(default)]
    pub projector_type: Option<String>,
    #[serde(default)]
    pub enable_confidence_head: Option<bool>,
    #[serde(default)]
    pub confidence_head_with_markov: Option<bool>,
    /// Some experimental EAGLE-3.1 exporters normalize every captured target
    /// layer independently before concatenation and `fc`. Atlas does not yet
    /// load or execute those per-capture norms, so such checkpoints must fail
    /// closed instead of silently serving a different computation graph.
    #[serde(default)]
    pub fc_layernorm: bool,
    /// DFlash 2 two-tap dynamic grouped causal conv (incoai/Qwen3.8-27B-DFlash2).
    /// Number of taps in `GroupedDynamicCausalConv` (2 for the released
    /// checkpoint). When > 1, every drafter layer carries an `attention_conv`
    /// and an `mlp_conv`, each with `base_kernel [2, kernel_size, H]` and
    /// `kernel_projection [2 * kernel_size * groups, H]` where
    /// `groups = H / conv_group_size`.
    #[serde(default)]
    pub conv_kernel_size: usize,
    /// DFlash 2 conv channel group — every `conv_group_size` channels share
    /// one dynamic coefficient (16 for the released checkpoint).
    #[serde(default)]
    pub conv_group_size: usize,
    /// DFlash 2 candidate-selector embedding rank (256 for the released
    /// checkpoint). `candidate_selector.hidden_projection` is
    /// `[rank, H]`; both codebooks are `[vocab, rank]`.
    #[serde(default)]
    pub selector_rank: usize,
    /// DFlash 2 candidate-selector beam width — top-k candidates per
    /// position scored by the codebook walk (16 for the released
    /// checkpoint).
    #[serde(default)]
    pub selector_top_k: usize,
    /// Logical vocabulary considered by the DFlash 2 candidate selector.
    ///
    /// The codebooks retain the physical padded vocabulary, while proposal
    /// top-k must crop logits to this value so padded rows are never
    /// candidates. `None` preserves every existing non-native checkpoint.
    #[serde(default)]
    pub selector_vocab_size: Option<usize>,
    /// Nested keys Atlas does not understand. Generic and V3 checkpoints keep
    /// their historical permissive parsing; exact native DFlash2 admission
    /// rejects this map unless it is empty so future trained semantics cannot
    /// be ignored silently.
    #[serde(default, flatten)]
    pub unknown_fields: std::collections::BTreeMap<String, serde_json::Value>,
}

/// Raw weight bundle for the DFlash drafter, post-load.
///
/// Verified against `z-lab/Qwen3.6-35B-A3B-DFlash` (commit 42d3b34, May 2026):
/// the checkpoint ships 91 BF16 tensors — `fc.weight`, `hidden_norm.weight`,
/// `norm.weight`, plus 11 weights per drafter layer × 8 layers. **No
/// `embed_tokens` or `lm_head` are in the checkpoint** — the drafter shares
/// the target's embedding and LM head at construction time. This matches the
/// vLLM PR #40898 flow: when those keys are absent, vLLM's `AutoWeightsLoader`
/// adds them to `skip_substrs`, leaving the runtime to slot in the target's
/// pointers.
#[allow(dead_code)]
pub struct DflashWeights {
    pub config: DflashConfig,

    /// `[draft_hidden, len(target_layer_ids) * target_hidden]`.
    /// Qwen3.6-35B-A3B-DFlash: `[2048, 10240]`.
    pub fc: DenseWeight,
    /// Serialized FC geometry retained for fail-closed target pairing.
    pub fc_shape: [usize; 2],
    /// `[draft_hidden]` — RMSNorm applied to the projected target context
    /// before mixing with token embeddings.
    pub hidden_norm: DenseWeight,
    /// `[draft_hidden]` — final RMSNorm before LM head.
    pub norm: DenseWeight,

    pub layers: Vec<DflashLayerWeights>,

    /// Present iff the drafter has a draft-id → target-id mapping (i.e.
    /// `draft_vocab_size != target_vocab_size`). Absent for
    /// Qwen3.6-35B-A3B-DFlash (both vocabs = 248320).
    pub draft_id_to_target_id: Option<Vec<i64>>,

    /// DSpark VanillaMarkov head weights, present iff the checkpoint ships
    /// `markov_head.markov_w1.weight` + `markov_head.markov_w2.weight` AND
    /// `config.markov_rank > 0`. `None` for the vanilla DFlash drafters.
    pub markov: Option<MarkovWeights>,

    /// DSpark learned acceptance-confidence projection. In the only supported
    /// planner mode (`static`) Atlas validates and loads these tensors for
    /// checkpoint integrity, but deliberately does not consume their output:
    /// this matches SGLang PR 34966's static verify-all path. Dynamic modes
    /// fail closed before weight loading until ragged verify and SPS costing
    /// are implemented.
    pub confidence: Option<DsparkConfidenceWeights>,

    /// DFlash 2 candidate selector (3 tensors), present iff the checkpoint
    /// ships `candidate_selector.*` AND `config.dflash_config` declares
    /// `selector_rank > 0`. `None` for plain DFlash / DSpark drafters.
    pub selector: Option<DflashSelectorWeights>,

    /// Native DSpine (`arXiv:2609.36173`) code-bridge weights, present iff the
    /// checkpoint declares `dflash_config.selected_prefix_code_bridge` and
    /// passes fail-closed version and zero-terminal checks.
    pub dspine: Option<DspineBridgeWeights>,
}

/// Loaded device weights for a DSpine `selected_prefix_bridge.*` checkpoint.
#[derive(Debug, Clone)]
pub struct DspineBridgeWeights {
    pub spec: DspineSpec,
    /// `[vocab_size, code_size]` BF16 token codebook.
    pub codebook: DenseWeight,
    /// `[code_size, hidden_size]` BF16 final-layer read projection.
    pub read_weight: DenseWeight,
    /// `[num_layers - 1, code_size, hidden_size]` BF16 early-layer read
    /// projections (`None` only when `num_layers == 1`).
    pub early_read_weights: Option<DenseWeight>,
    /// `[num_layers, hidden_size, rank]` BF16 per-layer residual write matrices.
    pub feature_write_weights: DenseWeight,
    /// `[rank, code_size]` BF16 receiver query projection (`gate.weight`).
    pub gate: DenseWeight,
    /// `[rank]` BF16 receiver gate bias (`gate.bias`).
    pub gate_bias: DenseWeight,
    /// `[rank, code_size]` BF16 predecessor sender gate (`predecessor_gate.weight`).
    pub predecessor_gate: DenseWeight,
    /// `[rank, code_size]` BF16 predecessor value projection (`prev.weight`).
    pub prev: DenseWeight,
    /// Optional `[vocab_size, terminal_residual_rank]` BF16 embedding.
    pub terminal_residual_embedding: Option<DenseWeight>,
    /// Optional `[rank, terminal_residual_rank]` BF16 projection.
    pub terminal_residual_projection: Option<DenseWeight>,
}

/// DSpark VanillaMarkov head weights (both BF16, loaded verbatim on device).
///
/// `markov_w1` is an `nn.Embedding(V, r)` — the checkpoint stores it as
/// `[V, r]` so a row `markov_w1[prev]` is a contiguous `[r]` slice (a plain
/// embedding gather). `markov_w2` is an `nn.Linear(r, V, bias=False)` — the
/// checkpoint stores its weight as `[V, r]` (out=V, in=r), i.e. the per-token
/// bias is `B(prev) = markov_w2 @ markov_w1[prev]` = `[V, r] @ [r] = [V]`,
/// which is exactly a `dense_gemv(input=w1row[r], weight=w2[V, r], n=V, k=r)`.
#[derive(Debug, Clone, Copy)]
pub struct MarkovWeights {
    /// `nn.Embedding(vocab, rank)` weight — `[vocab, rank]` BF16.
    pub w1: DenseWeight,
    /// `nn.Linear(rank, vocab, bias=False)` weight — `[vocab, rank]` BF16.
    pub w2: DenseWeight,
    /// Low-rank dimension `r` (256 for DSpark-AEON-draft).
    pub rank: usize,
}

/// Learned DSpark confidence affine (`sigmoid([hidden; markov] @ W^T + b)`).
#[derive(Debug, Clone, Copy)]
pub struct DsparkConfidenceWeights {
    /// `nn.Linear(input_dim, 1)` weight — `[1, input_dim]` BF16.
    pub proj: DenseWeight,
    /// Scalar bias — `[1]` BF16.
    pub bias: DenseWeight,
    /// `hidden_size + markov_rank` when `with_markov`, else `hidden_size`.
    pub input_dim: usize,
    pub with_markov: bool,
}

/// DFlash 2 `GroupedDynamicCausalConv` weights (one per sublayer).
///
/// `base_kernel` is `[2, kernel_size, hidden]` (stage 0 = prepare,
/// stage 1 = finish) and `kernel_projection` is
/// `[2 * kernel_size * groups, hidden]` with `groups = hidden / group_size`.
/// Both are BF16 in the released checkpoint.
#[derive(Debug, Clone)]
pub struct DflashConvWeights {
    /// `[2, kernel_size, hidden]` BF16.
    pub base_kernel: DenseWeight,
    /// `[2 * kernel_size * groups, hidden]` BF16.
    pub kernel_projection: DenseWeight,
    /// `kernel_size` (2 for the released checkpoint).
    pub kernel_size: usize,
    /// `hidden / conv_group_size`.
    pub groups: usize,
    /// `conv_group_size` (16 for the released checkpoint).
    pub group_size: usize,
}

/// DFlash 2 candidate-selector weights.
///
/// `hidden_projection` is `[rank, hidden]`; `predecessor_codebook` and
/// `successor_codebook` are both `[vocab, rank]` BF16 embedding tables.
#[derive(Debug, Clone)]
pub struct DflashSelectorWeights {
    /// `[rank, hidden]` BF16.
    pub hidden_projection: DenseWeight,
    /// `[vocab, rank]` BF16.
    pub predecessor_codebook: DenseWeight,
    /// `[vocab, rank]` BF16.
    pub successor_codebook: DenseWeight,
    /// Selector embedding rank (256 for the released checkpoint).
    pub rank: usize,
    /// Selector beam width — top-k candidates per position (16).
    pub top_k: usize,
}

/// Projection payload for one drafter layer. A layer is wholly BF16 or wholly
/// pre-quantized; mixed precision within the seven projections is invalid.
#[allow(dead_code)]
pub enum DflashLayerProjections {
    Bf16 {
        q_proj: DenseWeight,
        k_proj: DenseWeight,
        v_proj: DenseWeight,
        o_proj: DenseWeight,
        gate_proj: DenseWeight,
        up_proj: DenseWeight,
        down_proj: DenseWeight,
    },
    ModelOptNvfp4 {
        q_proj: QuantizedWeight,
        k_proj: QuantizedWeight,
        v_proj: QuantizedWeight,
        o_proj: QuantizedWeight,
        gate_proj: QuantizedWeight,
        up_proj: QuantizedWeight,
        down_proj: QuantizedWeight,
    },
}

/// Per-drafter-layer weights. Norms and DFlash2 convolutions remain BF16;
/// only the seven large projections may use direct ModelOpt NVFP4 payloads.
#[allow(dead_code)]
pub struct DflashLayerWeights {
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    pub projections: DflashLayerProjections,
    /// DFlash 2 two-tap conv wrapping the attention sublayer. `None` for
    /// non-DFlash2 drafters.
    pub attention_conv: Option<DflashConvWeights>,
    /// DFlash 2 two-tap conv wrapping the MLP sublayer. `None` for
    /// non-DFlash2 drafters.
    pub mlp_conv: Option<DflashConvWeights>,
}

/// Probe a [`WeightStore`] for the presence of DFlash drafter weights.
/// Returns true if the store contains the unique `fc.weight` tensor that
/// DFlash drafters ship — a lightweight detection that doesn't load any
/// data. Both bare-key and `model.`-prefixed layouts are accepted; the
/// canonical `z-lab/Qwen3.6-{27B,35B-A3B}-DFlash` checkpoints ship the
/// bare layout (verified against commit 42d3b34, May 2026).
pub fn store_has_dflash_weights(store: &WeightStore) -> bool {
    store.contains("fc.weight") || store.contains("model.fc.weight")
}

/// Probe a [`WeightStore`] for the DSpark VanillaMarkov head tensors. Both
/// bare and `model.`-prefixed layouts are accepted (the DSpark-AEON-draft
/// checkpoint ships bare: `markov_head.markov_w1.weight`).
pub fn store_has_markov_head(store: &WeightStore) -> bool {
    store.contains("markov_head.markov_w1.weight")
        || store.contains("model.markov_head.markov_w1.weight")
}

/// Parse a DFlash drafter's `config.json` into a [`DflashConfig`]. Used by
/// `main.rs` after fetching the drafter's HF metadata to size the runtime
/// `BlockDiffusionDraftHead` (layer count, head_dim, vocab_size, the
/// `target_layer_ids` capture indices).
pub fn parse_dflash_config(json: &str) -> Result<DflashConfig> {
    let raw: serde_json::Value =
        serde_json::from_str(json).context("Parsing DFlash drafter config.json")?;
    let root_block_size_explicit = raw
        .as_object()
        .is_some_and(|object| object.contains_key("block_size"));
    let root_second_pass = raw
        .as_object()
        .and_then(|object| object.get("code_bridge_second_pass"))
        .filter(|v| !v.is_null())
        .cloned();
    let mut config: DflashConfig =
        serde_json::from_value(raw).context("Parsing DFlash drafter config.json")?;
    config.root_block_size_explicit = root_block_size_explicit;
    match (
        config.dflash_config.as_ref(),
        config.mask_token_id,
        config.target_layer_ids.as_ref(),
    ) {
        (Some(sub), top_mask, top_layers) => {
            if let Some(mask) = top_mask {
                anyhow::ensure!(
                    mask == sub.mask_token_id,
                    "DFlash config disagrees on `mask_token_id`: top-level={mask}, \
                     dflash_config={}",
                    sub.mask_token_id
                );
            }
            if let Some(layers) = top_layers {
                anyhow::ensure!(
                    layers == &sub.target_layer_ids,
                    "DFlash config disagrees on `target_layer_ids`: top-level={layers:?}, \
                     dflash_config={:?}",
                    sub.target_layer_ids
                );
            }
        }
        (None, Some(mask_token_id), Some(target_layer_ids)) => {
            config.dflash_config = Some(DflashSubConfig {
                block_size: config.root_block_size_explicit.then_some(config.block_size),
                mask_token_id,
                target_layer_ids: target_layer_ids.clone(),
                projector_type: None,
                enable_confidence_head: None,
                confidence_head_with_markov: None,
                fc_layernorm: false,
                conv_kernel_size: 0,
                conv_group_size: 0,
                selector_rank: 0,
                selector_top_k: 0,
                selector_vocab_size: None,
                unknown_fields: Default::default(),
            });
        }
        (None, Some(_), None) | (None, None, Some(_)) => {
            anyhow::bail!(
                "DFlash config declares only one of top-level `mask_token_id` / \
                 `target_layer_ids` without a nested `dflash_config`"
            );
        }
        (None, None, None) => {}
    }
    if let Some(second_pass) = root_second_pass {
        let sub = config.dflash_config.as_mut().ok_or_else(|| {
            anyhow::anyhow!("DSpine `code_bridge_second_pass` requires a `dflash_config` block")
        })?;
        sub.unknown_fields
            .insert(ROOT_DSPINE_SECOND_PASS_KEY.to_string(), second_pass);
    }
    if config
        .dflash_config
        .as_ref()
        .is_some_and(|sub| sub.fc_layernorm)
    {
        anyhow::bail!(
            "DFlash checkpoint requires dflash_config.fc_layernorm=true, but Flux does not yet \
             load or execute the per-capture pre_fc_norms weights"
        );
    }
    config.validate_supported_semantics()?;
    super::dflash_validation::validate_native_flash_next_config(&config)?;
    Ok(config)
}

/// Return the exact pre-upload BF16 manifest for any checkpoint claimed by a
/// `quorum` drafter profile (Flash-Next V3, DFlash2, or DSpark), or `None`
/// otherwise.
pub fn native_flash_next_finiteness_manifest(
    config: &DflashConfig,
) -> Result<Option<std::collections::BTreeMap<String, Vec<usize>>>> {
    super::dflash_validation::native_flash_next_finiteness_manifest(config)
}

/// Load DFlash drafter weights from a separate [`WeightStore`] pointing at
/// the drafter checkpoint.
///
/// The drafter ships its weights at the **root** of the safetensors file
/// (no `model.` prefix), in the same naming convention as a vanilla Qwen3
/// transformer minus `embed_tokens` and `lm_head`. Atlas's runtime fills
/// those two from the *target* model's embedding / LM head at construction
/// time — exactly mirroring vLLM's "absent in checkpoint → skip_substrs →
/// share with parent" flow.
///
/// The probed key list (verified against `z-lab/Qwen3.6-35B-A3B-DFlash`):
///
/// ```text
///   fc.weight                                              [H, 5*H_target]
///   hidden_norm.weight                                     [H]
///   norm.weight                                            [H]
///   layers.{0..L-1}.input_layernorm.weight                 [H]
///   layers.{0..L-1}.post_attention_layernorm.weight        [H]
///   layers.{0..L-1}.self_attn.q_proj.weight                [Q*Hd, H]
///   layers.{0..L-1}.self_attn.k_proj.weight                [Kv*Hd, H]
///   layers.{0..L-1}.self_attn.v_proj.weight                [Kv*Hd, H]
///   layers.{0..L-1}.self_attn.o_proj.weight                [H, Q*Hd]
///   layers.{0..L-1}.self_attn.q_norm.weight                [Hd]
///   layers.{0..L-1}.self_attn.k_norm.weight                [Hd]
///   layers.{0..L-1}.mlp.gate_proj.weight                   [I, H]
///   layers.{0..L-1}.mlp.up_proj.weight                     [I, H]
///   layers.{0..L-1}.mlp.down_proj.weight                   [H, I]
/// ```
///
/// where `H=2048`, `H_target=2048`, `Q=32`, `Kv=4`, `Hd=128`, `I=6144`,
/// `L=8` for Qwen3.6-35B-A3B-DFlash.
///
/// Under TP the drafter is replicated, not sharded — `tp_size>1` produces
/// the same per-rank result as `tp_size=1`. Memory cost: ~948 MB BF16
/// per rank, trivially below the 119 GB GB10 budget.
pub fn load_dflash_weights(
    drafter_store: &WeightStore,
    drafter_config: &DflashConfig,
    gpu: &dyn GpuBackend,
    _tp_size: usize,
) -> Result<Option<DflashWeights>> {
    let Some(admission) = validate_dflash_store(drafter_store, drafter_config)? else {
        tracing::debug!("DFlash drafter store has no `fc.weight` — skipping");
        return Ok(None);
    };
    let prefix = admission.prefix;

    let fc_name = format!("{prefix}fc.weight");
    let fc_tensor = drafter_store.get(&fc_name)?;
    anyhow::ensure!(
        fc_tensor.shape.len() == 2,
        "DFlash drafter fc.weight must be rank 2, got {:?}",
        fc_tensor.shape
    );
    let fc_shape = [fc_tensor.shape[0], fc_tensor.shape[1]];
    let fc = dense(drafter_store, &fc_name).context("DFlash drafter: load fc.weight")?;
    let hidden_norm = dense(drafter_store, &format!("{prefix}hidden_norm.weight"))
        .context("DFlash drafter: load hidden_norm.weight")?;
    let norm = dense(drafter_store, &format!("{prefix}norm.weight"))
        .context("DFlash drafter: load norm.weight")?;

    let layer_count = drafter_config.num_hidden_layers;
    let hidden = drafter_config.hidden_size;
    let dflash2 = drafter_config.is_dflash2();
    let conv_kernel = drafter_config
        .dflash_config
        .as_ref()
        .map(|c| c.conv_kernel_size)
        .unwrap_or(0);
    let conv_group = drafter_config
        .dflash_config
        .as_ref()
        .map(|c| c.conv_group_size)
        .unwrap_or(0);
    let q_out = drafter_config.num_attention_heads * drafter_config.head_dim;
    let kv_out = drafter_config.num_key_value_heads * drafter_config.head_dim;
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        let lp = format!("{prefix}layers.{i}");
        // DFlash 2 two-tap conv: `layers.{i}.attention_conv.base_kernel`
        // `[2, kernel, hidden]` + `kernel_projection.weight`
        // `[2*kernel*groups, hidden]`, same for `mlp_conv`.
        let attention_conv = if dflash2 {
            let base_kernel = dense(drafter_store, &format!("{lp}.attention_conv.base_kernel"))
                .context("DFlash2 drafter: load attention_conv.base_kernel")?;
            let kernel_projection = dense(
                drafter_store,
                &format!("{lp}.attention_conv.kernel_projection.weight"),
            )
            .context("DFlash2 drafter: load attention_conv.kernel_projection.weight")?;
            let groups = hidden / conv_group.max(1);
            Some(DflashConvWeights {
                base_kernel,
                kernel_projection,
                kernel_size: conv_kernel.max(1),
                groups,
                group_size: conv_group.max(1),
            })
        } else {
            None
        };
        let mlp_conv = if dflash2 {
            let base_kernel = dense(drafter_store, &format!("{lp}.mlp_conv.base_kernel"))
                .context("DFlash2 drafter: load mlp_conv.base_kernel")?;
            let kernel_projection = dense(
                drafter_store,
                &format!("{lp}.mlp_conv.kernel_projection.weight"),
            )
            .context("DFlash2 drafter: load mlp_conv.kernel_projection.weight")?;
            let groups = hidden / conv_group.max(1);
            Some(DflashConvWeights {
                base_kernel,
                kernel_projection,
                kernel_size: conv_kernel.max(1),
                groups,
                group_size: conv_group.max(1),
            })
        } else {
            None
        };
        let projections = match admission.projection_format {
            super::dflash_validation::DflashProjectionFormat::Bf16 => {
                DflashLayerProjections::Bf16 {
                    q_proj: dense(drafter_store, &format!("{lp}.self_attn.q_proj.weight"))?,
                    k_proj: dense(drafter_store, &format!("{lp}.self_attn.k_proj.weight"))?,
                    v_proj: dense(drafter_store, &format!("{lp}.self_attn.v_proj.weight"))?,
                    o_proj: dense(drafter_store, &format!("{lp}.self_attn.o_proj.weight"))?,
                    gate_proj: dense(drafter_store, &format!("{lp}.mlp.gate_proj.weight"))?,
                    up_proj: dense(drafter_store, &format!("{lp}.mlp.up_proj.weight"))?,
                    down_proj: dense(drafter_store, &format!("{lp}.mlp.down_proj.weight"))?,
                }
            }
            super::dflash_validation::DflashProjectionFormat::ModelOptNvfp4 => {
                DflashLayerProjections::ModelOptNvfp4 {
                    q_proj: quantized_modelopt_with_shape(
                        drafter_store,
                        &format!("{lp}.self_attn.q_proj"),
                        q_out,
                        hidden,
                        gpu,
                    )?,
                    k_proj: quantized_modelopt_with_shape(
                        drafter_store,
                        &format!("{lp}.self_attn.k_proj"),
                        kv_out,
                        hidden,
                        gpu,
                    )?,
                    v_proj: quantized_modelopt_with_shape(
                        drafter_store,
                        &format!("{lp}.self_attn.v_proj"),
                        kv_out,
                        hidden,
                        gpu,
                    )?,
                    o_proj: quantized_modelopt_with_shape(
                        drafter_store,
                        &format!("{lp}.self_attn.o_proj"),
                        hidden,
                        q_out,
                        gpu,
                    )?,
                    gate_proj: quantized_modelopt_with_shape(
                        drafter_store,
                        &format!("{lp}.mlp.gate_proj"),
                        drafter_config.intermediate_size,
                        hidden,
                        gpu,
                    )?,
                    up_proj: quantized_modelopt_with_shape(
                        drafter_store,
                        &format!("{lp}.mlp.up_proj"),
                        drafter_config.intermediate_size,
                        hidden,
                        gpu,
                    )?,
                    down_proj: quantized_modelopt_with_shape(
                        drafter_store,
                        &format!("{lp}.mlp.down_proj"),
                        hidden,
                        drafter_config.intermediate_size,
                        gpu,
                    )?,
                }
            }
        };
        let layer = DflashLayerWeights {
            input_layernorm: dense(drafter_store, &format!("{lp}.input_layernorm.weight"))?,
            post_attention_layernorm: dense(
                drafter_store,
                &format!("{lp}.post_attention_layernorm.weight"),
            )?,
            q_norm: dense(drafter_store, &format!("{lp}.self_attn.q_norm.weight"))?,
            k_norm: dense(drafter_store, &format!("{lp}.self_attn.k_norm.weight"))?,
            projections,
            attention_conv,
            mlp_conv,
        };
        layers.push(layer);
    }

    // DFlash 2 candidate selector: `candidate_selector.hidden_projection`
    // `[rank, hidden]` + both codebooks `[vocab, rank]`. Loaded only when the
    // config declares the selector contract AND the tensors are present.
    let selector = if dflash2 {
        let rank = drafter_config
            .dflash_config
            .as_ref()
            .map(|c| c.selector_rank)
            .unwrap_or(0);
        let top_k = drafter_config
            .dflash_config
            .as_ref()
            .map(|c| c.selector_top_k)
            .unwrap_or(0);
        let hidden_projection = dense(
            drafter_store,
            &format!("{prefix}candidate_selector.hidden_projection.weight"),
        )
        .context("DFlash2 drafter: load candidate_selector.hidden_projection.weight")?;
        let predecessor_codebook = dense(
            drafter_store,
            &format!("{prefix}candidate_selector.predecessor_codebook"),
        )
        .context("DFlash2 drafter: load candidate_selector.predecessor_codebook")?;
        let successor_codebook = dense(
            drafter_store,
            &format!("{prefix}candidate_selector.successor_codebook"),
        )
        .context("DFlash2 drafter: load candidate_selector.successor_codebook")?;
        Some(DflashSelectorWeights {
            hidden_projection,
            predecessor_codebook,
            successor_codebook,
            rank,
            top_k,
        })
    } else {
        None
    };

    // `d2t` / `draft_id_to_target_id` remapping for reduced-vocab drafters is
    // not yet implemented; fail closed instead of silently serving unmapped
    // draft token IDs.
    if drafter_store.contains(&format!("{prefix}d2t"))
        || drafter_store.contains(&format!("{prefix}draft_id_to_target_id"))
    {
        anyhow::bail!(
            "DFlash draft-id to target-id remapping (`d2t` / `draft_id_to_target_id`) is not yet \
             implemented"
        );
    }
    let draft_id_to_target_id = None;

    // DSpark VanillaMarkov head (config-driven, like the rest of the drafter):
    // load W1/W2 iff `config.markov_rank > 0` AND the checkpoint actually ships
    // the tensors. ATLAS_DFLASH_MARKOV=0 disables the head at load time even
    // when present (A/B toggle); default auto-on when the checkpoint has it.
    let markov_disabled = std::env::var("ATLAS_DFLASH_MARKOV").ok().as_deref() == Some("0");
    let markov = if drafter_config.markov_rank > 0
        && store_has_markov_head(drafter_store)
        && !markov_disabled
    {
        let head_type = drafter_config.markov_head_type.to_lowercase();
        let w1 = dense(
            drafter_store,
            &format!("{prefix}markov_head.markov_w1.weight"),
        )
        .context("DFlash Markov head: load markov_w1.weight")?;
        let w2 = dense(
            drafter_store,
            &format!("{prefix}markov_head.markov_w2.weight"),
        )
        .context("DFlash Markov head: load markov_w2.weight")?;
        tracing::info!(
            "DFlash DSpark Markov head loaded: rank={}, type={} (bigram logit bias applied \
             LEFT-TO-RIGHT before per-position argmax; set ATLAS_DFLASH_MARKOV=0 to disable)",
            drafter_config.markov_rank,
            head_type,
        );
        Some(MarkovWeights {
            w1,
            w2,
            rank: drafter_config.markov_rank,
        })
    } else {
        if drafter_config.markov_rank > 0 && !store_has_markov_head(drafter_store) {
            tracing::warn!(
                "DFlash config has markov_rank={} but the checkpoint ships no \
                 `markov_head.markov_w1.weight` — Markov head disabled",
                drafter_config.markov_rank,
            );
        }
        if markov_disabled && store_has_markov_head(drafter_store) {
            tracing::info!("DFlash Markov head present but ATLAS_DFLASH_MARKOV=0 — disabled");
        }
        None
    };

    // In SGLang PR 34966's default `static` mode the learned confidence head
    // is not constructed or evaluated: every proposal is verified. Atlas
    // still validates and loads the public checkpoint's tensors so corrupt or
    // mismatched exports cannot masquerade as supported. Dynamic planners are
    // rejected separately by `validate_verify_mode`.
    let confidence = if let Some(confidence) = drafter_config.confidence_head_config()? {
        let proj = dense(
            drafter_store,
            &format!("{prefix}confidence_head.proj.weight"),
        )
        .context("DSpark confidence head: load proj.weight")?;
        let bias = dense(drafter_store, &format!("{prefix}confidence_head.proj.bias"))
            .context("DSpark confidence head: load proj.bias")?;
        Some(DsparkConfidenceWeights {
            proj,
            bias,
            input_dim: confidence.input_dim,
            with_markov: confidence.with_markov,
        })
    } else {
        None
    };

    let dspine = load_dspine_bridge(drafter_store, prefix, drafter_config, gpu)?;

    tracing::info!(
        "Drafter weights loaded: family={}, {} layers, hidden={}, vocab={}, \
         checkpoint_block_size={}, target_layers={:?}, markov={}, confidence={}, \
         dflash2={} selector={} dspine={} conv_per_layer={} projection_format={:?}",
        drafter_config.checkpoint_family()?,
        layers.len(),
        drafter_config.hidden_size,
        drafter_config.vocab_size,
        drafter_config.resolved_block_size(),
        drafter_config
            .dflash_config
            .as_ref()
            .map(|c| c.target_layer_ids.as_slice())
            .unwrap_or(&[]),
        markov.is_some(),
        confidence.is_some(),
        dflash2,
        selector.is_some(),
        dspine.is_some(),
        layers.iter().filter(|l| l.attention_conv.is_some()).count(),
        admission.projection_format,
    );

    Ok(Some(DflashWeights {
        config: drafter_config.clone(),
        fc,
        fc_shape,
        hidden_norm,
        norm,
        layers,
        draft_id_to_target_id,
        markov,
        confidence,
        selector,
        dspine,
    }))
}

/// Fail-closed validation of a DSpine scalar version tensor (`I64` scalar `== 1`).
pub(crate) fn validate_dspine_version_bytes(
    name: &str,
    dtype: spark_runtime::weights::WeightDtype,
    shape: &[usize],
    bytes: &[u8],
) -> Result<()> {
    anyhow::ensure!(
        dtype == spark_runtime::weights::WeightDtype::Int64,
        "DSpine version tensor `{name}` must have dtype Int64, got {dtype:?}"
    );
    anyhow::ensure!(
        shape.is_empty() || shape == [1],
        "DSpine version tensor `{name}` must be a scalar ([] or [1]), got {shape:?}"
    );
    let arr: [u8; 8] = bytes.try_into().map_err(|_| {
        anyhow::anyhow!(
            "DSpine version tensor `{name}` expected 8 bytes, got {}",
            bytes.len()
        )
    })?;
    let val = i64::from_le_bytes(arr);
    anyhow::ensure!(
        val == 1,
        "Unsupported DSpine checkpoint version `{name}` = {val} (expected 1)"
    );
    Ok(())
}

/// Fail-closed validation that `selected_prefix_bridge.out.weight` is all-zero BF16
/// (standalone terminal readout disabled).
pub(crate) fn validate_dspine_zero_terminal_bf16(name: &str, bytes: &[u8]) -> Result<()> {
    anyhow::ensure!(
        bytes.len().is_multiple_of(2),
        "DSpine BF16 tensor `{name}` byte length {} is odd",
        bytes.len()
    );
    for (idx, chunk) in bytes.chunks_exact(2).enumerate() {
        let word = u16::from_le_bytes([chunk[0], chunk[1]]);
        anyhow::ensure!(
            (word & 0x7fff) == 0,
            "DSpine `{name}` element {idx} is non-zero (bf16 bits 0x{word:04x}); \
             standalone terminal readout is disabled in supported DSpine checkpoints"
        );
    }
    Ok(())
}

fn require_dspine_bf16(
    store: &WeightStore,
    name: &str,
    expected_shape: &[usize],
) -> Result<DenseWeight> {
    let meta = store
        .get(name)
        .with_context(|| format!("DSpine checkpoint missing required tensor `{name}`"))?;
    anyhow::ensure!(
        meta.dtype == spark_runtime::weights::WeightDtype::BF16,
        "DSpine tensor `{name}` has dtype {:?}; expected BF16",
        meta.dtype
    );
    anyhow::ensure!(
        meta.shape.as_slice() == expected_shape,
        "DSpine tensor `{name}` has shape {:?}; expected {expected_shape:?}",
        meta.shape
    );
    dense(store, name)
}

fn check_dspine_version_tensor(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    name: &str,
    required: bool,
) -> Result<()> {
    if !store.contains(name) {
        anyhow::ensure!(
            !required,
            "DSpine checkpoint missing required version tensor `{name}`"
        );
        return Ok(());
    }
    let meta = store.get(name)?;
    if meta.dtype != spark_runtime::weights::WeightDtype::Int64
        || !(meta.shape.is_empty() || meta.shape == [1])
    {
        return validate_dspine_version_bytes(name, meta.dtype, &meta.shape, &[0u8; 8]);
    }
    let mut raw = [0u8; 8];
    gpu.copy_d2h(meta.ptr, &mut raw)
        .with_context(|| format!("D2H read of DSpine version tensor `{name}`"))?;
    validate_dspine_version_bytes(name, meta.dtype, &meta.shape, &raw)
}

fn load_dspine_bridge(
    store: &WeightStore,
    prefix: &str,
    config: &DflashConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<DspineBridgeWeights>> {
    let bridge_prefix = format!("{prefix}selected_prefix_bridge.");
    let Some(spec) = config.dspine_spec()? else {
        let stray = store.names().find(|n| n.starts_with(&bridge_prefix));
        anyhow::ensure!(
            stray.is_none(),
            "Drafter checkpoint contains DSpine tensor {:?}, but `dflash_config.selected_prefix_code_bridge` \
             is absent; refusing to silently ignore trained bridge weights",
            stray.unwrap_or_default()
        );
        return Ok(None);
    };

    for forbidden_sub in ["prefix_gru.", "embed_proj.", "candidate_selector."] {
        let pat = format!("{prefix}{forbidden_sub}");
        if let Some(found) = store.names().find(|n| n.starts_with(&pat)) {
            anyhow::bail!(
                "DSpine checkpoint cannot carry Domino / DFlash2 selector weights (`{found}`)"
            );
        }
    }

    check_dspine_version_tensor(
        store,
        gpu,
        &format!("{bridge_prefix}independent_write_version"),
        true,
    )?;
    check_dspine_version_tensor(
        store,
        gpu,
        &format!("{bridge_prefix}terminal_disabled_version"),
        true,
    )?;
    check_dspine_version_tensor(
        store,
        gpu,
        &format!("{prefix}draft_block_causal_version"),
        false,
    )?;
    check_dspine_version_tensor(
        store,
        gpu,
        &format!("{prefix}last_write_refine_version"),
        false,
    )?;

    let out_name = format!("{bridge_prefix}out.weight");
    let out_weight = require_dspine_bf16(store, &out_name, &[spec.code_size, spec.rank])?;
    let mut out_bytes = vec![0u8; spec.code_size * spec.rank * 2];
    gpu.copy_d2h(out_weight.weight, &mut out_bytes)
        .with_context(|| format!("D2H read of `{out_name}`"))?;
    validate_dspine_zero_terminal_bf16(&out_name, &out_bytes)?;

    let vocab = config.vocab_size;
    let hidden = config.hidden_size;
    let num_layers = config.num_hidden_layers;
    let code_size = spec.code_size;
    let rank = spec.rank;

    let codebook = require_dspine_bf16(
        store,
        &format!("{bridge_prefix}codebook"),
        &[vocab, code_size],
    )?;
    let read_weight = require_dspine_bf16(
        store,
        &format!("{bridge_prefix}read_weight"),
        &[code_size, hidden],
    )?;
    let early_read_weights = if num_layers > 1 {
        Some(require_dspine_bf16(
            store,
            &format!("{bridge_prefix}early_read_weights"),
            &[num_layers - 1, code_size, hidden],
        )?)
    } else {
        None
    };
    let feature_write_weights = require_dspine_bf16(
        store,
        &format!("{bridge_prefix}feature_write_weights"),
        &[num_layers, hidden, rank],
    )?;
    let gate = require_dspine_bf16(
        store,
        &format!("{bridge_prefix}gate.weight"),
        &[rank, code_size],
    )?;
    let gate_bias = require_dspine_bf16(store, &format!("{bridge_prefix}gate.bias"), &[rank])?;
    let predecessor_gate = require_dspine_bf16(
        store,
        &format!("{bridge_prefix}predecessor_gate.weight"),
        &[rank, code_size],
    )?;
    let prev = require_dspine_bf16(
        store,
        &format!("{bridge_prefix}prev.weight"),
        &[rank, code_size],
    )?;

    let tr_emb_name = format!("{bridge_prefix}terminal_residual_embedding");
    let tr_proj_name = format!("{bridge_prefix}terminal_residual_projection");
    let (terminal_residual_embedding, terminal_residual_projection) =
        if spec.terminal_residual_rank > 0 {
            let rt = spec.terminal_residual_rank;
            (
                Some(require_dspine_bf16(store, &tr_emb_name, &[vocab, rt])?),
                Some(require_dspine_bf16(store, &tr_proj_name, &[rank, rt])?),
            )
        } else {
            anyhow::ensure!(
                !store.contains(&tr_emb_name) && !store.contains(&tr_proj_name),
                "DSpine checkpoint contains terminal_residual tensors when \
                 selected_prefix_terminal_residual_rank=0"
            );
            (None, None)
        };

    tracing::info!(
        "DSpine code bridge loaded: layers={}, code_size={}, rank={}, head_topk={}, \
         terminal_residual_rank={}, active_mask=0x{:x}",
        num_layers,
        code_size,
        rank,
        spec.head_topk,
        spec.terminal_residual_rank,
        spec.active_layers_mask,
    );

    Ok(Some(DspineBridgeWeights {
        spec,
        codebook,
        read_weight,
        early_read_weights,
        feature_write_weights,
        gate,
        gate_bias,
        predecessor_gate,
        prev,
        terminal_residual_embedding,
        terminal_residual_projection,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE_CONFIG_FIELDS: &str = r#"
        "hidden_size": 5120,
        "num_hidden_layers": 5,
        "intermediate_size": 10240,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 248320,
        "block_size": 7,
        "markov_rank": 256,
        "markov_head_type": "vanilla"
    "#;

    /// Smoke-test the DFlash drafter `config.json` parser against a local
    /// `z-lab/Qwen3.6-35B-A3B-DFlash` config when `FLUX_TEST_DFLASH_CONFIG`
    /// is set. Skipped when unset so CI stays hermetic. Asserts the drafter
    /// dimensions: 8 layers, hidden=2048, vocab=248320, B=16/γ=15,
    /// mask=248070, layer_ids=[1,10,19,28,37].
    #[test]
    fn parse_qwen3_6_35b_dflash_config() {
        let Ok(path) = std::env::var("FLUX_TEST_DFLASH_CONFIG") else {
            return;
        };
        let json = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: drafter config {path} not found");
                return;
            }
        };
        let config = parse_dflash_config(&json).expect("parse drafter config");
        assert_eq!(config.num_hidden_layers, 8);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.intermediate_size, 6144);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 248320);
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.block_size, 16);
        let sub = config.dflash_config.expect("dflash_config present");
        assert_eq!(sub.mask_token_id, 248070);
        assert_eq!(sub.target_layer_ids, vec![1, 10, 19, 28, 37]);
    }

    #[test]
    fn rejects_unimplemented_per_capture_fc_layernorm() {
        let json = r#"{
            "hidden_size": 5120,
            "num_hidden_layers": 6,
            "intermediate_size": 17408,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 248320,
            "dflash_config": {
                "mask_token_id": 248077,
                "target_layer_ids": [1, 10, 18, 27, 35, 44, 52, 61],
                "fc_layernorm": true
            }
        }"#;
        let err = parse_dflash_config(json).expect_err("fc_layernorm must fail closed");
        assert!(err.to_string().contains("fc_layernorm"));
    }

    #[test]
    fn resolves_specforge_dspark_only_from_agreeing_explicit_declarations() {
        let json = format!(
            r#"{{
                "architectures": ["DSparkDraftModel"],
                {BASE_CONFIG_FIELDS},
                "dflash_config": {{
                    "mask_token_id": 248077,
                    "target_layer_ids": [4, 16, 28, 40, 52],
                    "projector_type": "dspark"
                }}
            }}"#
        );
        let config = parse_dflash_config(&json).expect("supported headless DSpark config");
        assert_eq!(
            config.checkpoint_family().unwrap(),
            DrafterCheckpointFamily::Dspark
        );
        let gamma = config.resolve_draft_count(None).unwrap();
        assert_eq!(gamma, 7);
        assert_eq!(gamma + 1, 8);
        assert_eq!(config.resolve_draft_count(Some(4)).unwrap(), 4);
        assert_eq!(config.resolve_draft_count(Some(7)).unwrap(), 7);
        assert!(config.resolve_draft_count(Some(8)).is_err());
        assert!(config.resolve_draft_count(Some(0)).is_err());
        let mut one_draft = config;
        one_draft.block_size = 1;
        assert_eq!(one_draft.resolve_draft_count(None).unwrap(), 1);
    }

    #[test]
    fn dspark_two_x_width_is_explicit_bounded_and_family_specific() {
        assert!(draft_width_is_supported(
            DrafterCheckpointFamily::Dspark,
            7,
            14,
            true,
            false,
        ));
        assert!(!draft_width_is_supported(
            DrafterCheckpointFamily::Dspark,
            7,
            14,
            false,
            false,
        ));
        assert!(!draft_width_is_supported(
            DrafterCheckpointFamily::Dspark,
            7,
            15,
            true,
            false,
        ));
        assert!(!draft_width_is_supported(
            DrafterCheckpointFamily::Dflash,
            7,
            14,
            true,
            false,
        ));
        assert!(draft_width_is_supported(
            DrafterCheckpointFamily::Dflash,
            7,
            7,
            false,
            false,
        ));
        assert!(draft_width_is_supported(
            DrafterCheckpointFamily::Dflash,
            7,
            15,
            false,
            true,
        ));
        assert!(!draft_width_is_supported(
            DrafterCheckpointFamily::Dflash,
            7,
            15,
            false,
            false,
        ));
        assert!(!draft_width_is_supported(
            DrafterCheckpointFamily::Dflash,
            7,
            16,
            false,
            true,
        ));
        assert!(!draft_width_is_supported(
            DrafterCheckpointFamily::Dflash,
            15,
            31,
            false,
            true,
        ));
    }

    #[test]
    fn preserves_legacy_dflash_family_without_dspark_markers() {
        let json = format!(
            r#"{{
                "architectures": ["DFlashDraftModel"],
                {BASE_CONFIG_FIELDS},
                "dflash_config": {{
                    "mask_token_id": 248070,
                    "target_layer_ids": [1, 10, 19, 28, 37]
                }}
            }}"#
        );
        let config = parse_dflash_config(&json).expect("legacy DFlash config");
        assert_eq!(config.resolved_rms_norm_eps().to_bits(), 1e-6f32.to_bits());
        assert_eq!(config.resolved_hidden_act(), "silu");
        assert_eq!(
            config.checkpoint_family().unwrap(),
            DrafterCheckpointFamily::Dflash
        );
        config
            .validate_verify_mode(DsparkVerifyMode::Static)
            .expect("static preserves legacy DFlash");
        let error = config
            .validate_verify_mode(DsparkVerifyMode::Compact)
            .expect_err("DSpark dynamic planner must not be applied to DFlash");
        assert!(error.to_string().contains("DFlash checkpoint"), "{error:#}");
        let mut b16 = config;
        b16.block_size = 16;
        assert_eq!(b16.resolve_draft_count(None).unwrap(), 15);
        assert_eq!(b16.resolve_draft_count(Some(4)).unwrap(), 4);
        assert!(b16.resolve_draft_count(Some(16)).is_err());
        assert!(b16.resolve_draft_count(Some(0)).is_err());
        b16.block_size = 1;
        assert!(b16.resolve_draft_count(None).is_err());
    }

    #[test]
    fn rejects_ambiguous_dspark_family_markers() {
        let architecture_only = format!(
            r#"{{
                "architectures": ["DSparkDraftModel"],
                {BASE_CONFIG_FIELDS},
                "dflash_config": {{
                    "mask_token_id": 248077,
                    "target_layer_ids": [4, 16, 28, 40, 52]
                }}
            }}"#
        );
        let projector_only = format!(
            r#"{{
                "architectures": ["DFlashDraftModel"],
                {BASE_CONFIG_FIELDS},
                "dflash_config": {{
                    "mask_token_id": 248077,
                    "target_layer_ids": [4, 16, 28, 40, 52],
                    "projector_type": "dspark"
                }}
            }}"#
        );
        for json in [architecture_only, projector_only] {
            let error = parse_dflash_config(&json).expect_err("ambiguous family must fail closed");
            assert!(
                error.to_string().contains("DSparkDraftModel")
                    || error.to_string().contains("projector_type"),
                "{error:#}"
            );
        }
    }

    #[test]
    fn resolves_public_dspark_confidence_contract_without_silent_fallback() {
        let json = format!(
            r#"{{
                "architectures": ["DSparkDraftModel"],
                "enable_confidence_head": true,
                "confidence_head_with_markov": true,
                {BASE_CONFIG_FIELDS},
                "dflash_config": {{
                    "mask_token_id": 248077,
                    "target_layer_ids": [4, 16, 28, 40, 52],
                    "projector_type": "dspark",
                    "enable_confidence_head": true,
                    "confidence_head_with_markov": true
                }}
            }}"#
        );
        let config = parse_dflash_config(&json).expect("public DSpark confidence config");
        assert_eq!(
            config.confidence_head_config().unwrap(),
            Some(DsparkConfidenceConfig {
                with_markov: true,
                input_dim: 5376,
            })
        );
        config
            .validate_verify_mode(DsparkVerifyMode::Static)
            .expect("static verify-all is supported");
        for mode in [DsparkVerifyMode::CapAccept, DsparkVerifyMode::Compact] {
            let error = config
                .validate_verify_mode(mode)
                .expect_err("dynamic mode must fail closed");
            assert!(error.to_string().contains("ragged verify"), "{error:#}");
            assert!(error.to_string().contains("SPS"), "{error:#}");
        }
    }

    #[test]
    fn rejects_confidence_config_disagreement_and_invalid_markov_dependency() {
        let disagreement = format!(
            r#"{{
                "architectures": ["DSparkDraftModel"],
                "enable_confidence_head": true,
                {BASE_CONFIG_FIELDS},
                "dflash_config": {{
                    "mask_token_id": 248077,
                    "target_layer_ids": [4, 16, 28, 40, 52],
                    "projector_type": "dspark",
                    "enable_confidence_head": false
                }}
            }}"#
        );
        let error = parse_dflash_config(&disagreement).expect_err("must not select one value");
        assert!(error.to_string().contains("disagrees"), "{error:#}");

        let no_head = format!(
            r#"{{
                "architectures": ["DSparkDraftModel"],
                "enable_confidence_head": false,
                "confidence_head_with_markov": true,
                {BASE_CONFIG_FIELDS},
                "dflash_config": {{
                    "mask_token_id": 248077,
                    "target_layer_ids": [4, 16, 28, 40, 52],
                    "projector_type": "dspark"
                }}
            }}"#
        );
        let error = parse_dflash_config(&no_head).expect_err("Markov feature needs head");
        assert!(
            error
                .to_string()
                .contains("requires enable_confidence_head=true"),
            "{error:#}"
        );
    }

    #[test]
    fn dspark_verify_mode_parser_never_falls_back() {
        assert_eq!(
            "static".parse::<DsparkVerifyMode>().unwrap(),
            DsparkVerifyMode::Static
        );
        assert_eq!(
            "cap-accept".parse::<DsparkVerifyMode>().unwrap(),
            DsparkVerifyMode::CapAccept
        );
        assert_eq!(
            "compact".parse::<DsparkVerifyMode>().unwrap(),
            DsparkVerifyMode::Compact
        );
        assert!("dynamic".parse::<DsparkVerifyMode>().is_err());
        assert!("STATIC".parse::<DsparkVerifyMode>().is_err());
    }

    #[test]
    fn parses_qwen38_27b_and_qwen3_8b_dspine_b16_configs() {
        let qwen38_27b_dspine = r#"{
            "architectures": ["DSpineDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 5120,
            "num_hidden_layers": 5,
            "num_target_layers": 64,
            "intermediate_size": 17408,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 248320,
            "block_size": 16,
            "is_causal": true,
            "sliding_window": 8192,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention"
            ],
            "code_bridge_second_pass": {
                "mode": "serial_table",
                "first_exact": 4,
                "head_topk": 16
            },
            "dflash_config": {
                "block_size": 16,
                "mask_token_id": 248070,
                "target_layer_ids": [5, 19, 33, 47, 61],
                "draft_block_causal": true,
                "selected_prefix_terminal": false,
                "selected_prefix_feature_writes": true,
                "selected_prefix_feature_write_mode": "independent",
                "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
                "selected_prefix_feature_next_token_rows": false,
                "selected_prefix_feature_gt_training": true,
                "selected_prefix_terminal_residual_rank": 64,
                "selected_prefix_code_bridge": {
                    "code_size": 256,
                    "rank": 256,
                    "eps": 1e-6
                },
                "selected_prefix_last_write_refine": {
                    "kind": "exact_sender",
                    "version": 1,
                    "head_topk": 16,
                    "weight": 1.0
                }
            }
        }"#;
        let cfg27 = parse_dflash_config(qwen38_27b_dspine).unwrap();
        assert!(cfg27.is_dspine());
        assert!(cfg27.is_draft_block_causal());
        assert_eq!(cfg27.resolve_draft_count(None).unwrap(), 15);
        let spec27 = cfg27.dspine_spec().unwrap().unwrap();
        assert_eq!(spec27.code_size, 256);
        assert_eq!(spec27.rank, 256);
        assert_eq!(spec27.head_topk, 16);
        assert_eq!(spec27.active_layers_mask, 0b11111);
        assert_eq!(spec27.terminal_residual_rank, 64);

        let qwen3_8b_dspine = r#"{
            "architectures": ["DSpineDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 4096,
            "num_hidden_layers": 5,
            "num_target_layers": 36,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "block_size": 16,
            "dflash_config": {
                "mask_token_id": 151669,
                "target_layer_ids": [2, 10, 18, 26, 34],
                "draft_block_causal": true,
                "selected_prefix_terminal": false,
                "selected_prefix_feature_writes": true,
                "selected_prefix_feature_write_mode": "independent",
                "selected_prefix_feature_layers": [2, 5],
                "selected_prefix_code_bridge": {
                    "code_size": 128,
                    "rank": 256
                },
                "selected_prefix_last_write_refine": {
                    "kind": "exact_sender",
                    "version": 1,
                    "head_topk": 16
                }
            }
        }"#;
        let cfg8 = parse_dflash_config(qwen3_8b_dspine).unwrap();
        assert!(cfg8.is_dspine());
        assert!(cfg8.is_draft_block_causal());
        let spec8 = cfg8.dspine_spec().unwrap().unwrap();
        assert_eq!(spec8.code_size, 128);
        assert_eq!(spec8.rank, 256);
        assert_eq!(spec8.head_topk, 16);
        assert_eq!(spec8.active_layers_mask, (1 << 1) | (1 << 4));
        assert!(!spec8.is_layer_active(0));
        assert!(spec8.is_layer_active(1));
        assert!(spec8.is_layer_active(4));
    }

    #[test]
    fn dspine_config_and_tensor_validators_fail_closed() {
        use spark_runtime::weights::WeightDtype;

        // Version scalar checks: Int64 scalar == 1 passes; wrong version, dtype, or shape fails.
        validate_dspine_version_bytes("v", WeightDtype::Int64, &[], &1i64.to_le_bytes()).unwrap();
        validate_dspine_version_bytes("v", WeightDtype::Int64, &[1], &1i64.to_le_bytes()).unwrap();
        assert!(
            validate_dspine_version_bytes("v", WeightDtype::Int64, &[], &2i64.to_le_bytes())
                .is_err()
        );
        assert!(
            validate_dspine_version_bytes("v", WeightDtype::Int64, &[], &0i64.to_le_bytes())
                .is_err()
        );
        assert!(
            validate_dspine_version_bytes("v", WeightDtype::BF16, &[], &1i64.to_le_bytes())
                .is_err()
        );
        assert!(
            validate_dspine_version_bytes("v", WeightDtype::Int64, &[2], &1i64.to_le_bytes())
                .is_err()
        );

        // Zero terminal `out.weight` check: +0.0 and -0.0 pass; any non-zero or NaN fails.
        validate_dspine_zero_terminal_bf16("out.weight", &[0, 0, 0x00, 0x80]).unwrap();
        assert!(validate_dspine_zero_terminal_bf16("out.weight", &[0, 0, 0x80, 0x3f]).is_err());
        assert!(validate_dspine_zero_terminal_bf16("out.weight", &[0x01, 0x7f]).is_err());

        // Non-causal DSpine config fails closed.
        let non_causal = r#"{
            "architectures": ["DSpineDraftModel"],
            "hidden_size": 4096,
            "num_hidden_layers": 5,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "block_size": 16,
            "is_causal": false,
            "dflash_config": {
                "mask_token_id": 151669,
                "target_layer_ids": [2, 10, 18, 26, 34],
                "selected_prefix_code_bridge": { "code_size": 128, "rank": 256 },
                "selected_prefix_last_write_refine": { "kind": "exact_sender", "version": 1, "head_topk": 16 }
            }
        }"#;
        assert!(parse_dflash_config(non_causal).is_err());

        // Disagreed head_topk between root second_pass and nested refine fails closed.
        let disagreed_topk = r#"{
            "architectures": ["DSpineDraftModel"],
            "hidden_size": 4096,
            "num_hidden_layers": 5,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "block_size": 16,
            "is_causal": true,
            "code_bridge_second_pass": { "mode": "serial_table", "first_exact": 4, "head_topk": 8 },
            "dflash_config": {
                "mask_token_id": 151669,
                "target_layer_ids": [2, 10, 18, 26, 34],
                "selected_prefix_code_bridge": { "code_size": 128, "rank": 256 },
                "selected_prefix_last_write_refine": { "kind": "exact_sender", "version": 1, "head_topk": 16 }
            }
        }"#;
        assert!(parse_dflash_config(disagreed_topk).is_err());

        // Missing final layer in selected_prefix_feature_layers fails closed.
        let missing_last = r#"{
            "architectures": ["DSpineDraftModel"],
            "hidden_size": 4096,
            "num_hidden_layers": 5,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "block_size": 16,
            "is_causal": true,
            "dflash_config": {
                "mask_token_id": 151669,
                "target_layer_ids": [2, 10, 18, 26, 34],
                "selected_prefix_feature_layers": [1, 2, 3, 4],
                "selected_prefix_code_bridge": { "code_size": 128, "rank": 256 },
                "selected_prefix_last_write_refine": { "kind": "exact_sender", "version": 1, "head_topk": 16 }
            }
        }"#;
        assert!(parse_dflash_config(missing_last).is_err());
    }
}
