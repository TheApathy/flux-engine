// SPDX-License-Identifier: AGPL-3.0-only

//! Weight-store loading: main checkpoint, prefix auto-detect, DFlash drafter.

use std::path::Path;

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

/// Measured resident-over-requested ratio of the NVFP4 weight store.
///
/// Both loaders `cuMemAlloc` one allocation per tensor (the fast loader:
/// `fast_weights/mod.rs`; the slow loader: `weights/loader.rs`), and on large
/// MoE checkpoints that produces ~150k allocations whose allocator rounding is
/// carried until the unified MoE transpose pass consolidates each layer.
/// Measured resident device memory delta over requested bytes is ~1.165x
/// (~90.3 GB vs ~77.5 GB requested). The previous 1.02 assumed an mmap'd
/// zero-copy store that neither loader implements.
pub(crate) const NVFP4_PER_TENSOR_ALLOC_MULTIPLIER: f64 = 1.165;

pub(crate) fn quant_multiplier(config: &ModelConfig) -> Option<f64> {
    if config.model_type == "minimax_m2" {
        return Some(1.02);
    }
    let qc = config.quantization_config.as_ref()?;
    if qc.quant_method == "fp8" {
        return Some(1.05);
    }
    // NVFP4: every tensor is a separate device allocation, so the *store*
    // costs its on-disk bytes plus per-allocation rounding — the measured
    // `NVFP4_PER_TENSOR_ALLOC_MULTIPLIER`. This ratio covers only the store
    // and its allocator overhead. Everything the model builder retains on
    // top of the store is architecture-dependent and is estimated in
    // absolute bytes by `construction_overhead_bytes` below — do NOT fold
    // it into this constant. A blanket ratio big enough for a hybrid
    // linear-attention model false-OOMs a plain one (e.g., a 1.3x blanket
    // ratio false-OOMs a 76 GB 122B checkpoint on a 119 GB GPU).
    // ModelOpt 0.47 MIXED_PRECISION sidecars sometimes label the aggregate
    // `quant_algo` "MIXED_PRECISION" even when every typed group it lists
    // is NVFP4/MXFP8 — identical on-disk tensor format to a checkpoint
    // that labels itself "NVFP4" directly. Treat the two the same *only*
    // when `config_groups` is non-empty and unanimous; an actually-mixed
    // checkpoint (some groups FP8, or no typed groups at all) must keep
    // falling through to the 1.3x heuristic below, avoiding ~22 GB of
    // false-OOM overestimation on NVFP4-labeled mixed-precision sidecars.
    let is_uniform_nvfp4_mixed_precision = qc.quant_algo.eq_ignore_ascii_case("MIXED_PRECISION")
        && !qc.config_groups.is_empty()
        && qc.config_groups.iter().all(|group| {
            matches!(
                group.weight_format,
                atlas_core::config::ModelOptWeightFormat::Nvfp4
                    | atlas_core::config::ModelOptWeightFormat::Mxfp8
            )
        });
    let is_nvfp4 = (qc.quant_method == "modelopt"
        && (qc.quant_algo.eq_ignore_ascii_case("NVFP4") || is_uniform_nvfp4_mixed_precision))
        || qc.quant_method == "compressed-tensors";
    if is_nvfp4 {
        return Some(NVFP4_PER_TENSOR_ALLOC_MULTIPLIER);
    }
    None
}

#[cfg(test)]
mod quant_multiplier_tests {
    use super::{NVFP4_PER_TENSOR_ALLOC_MULTIPLIER, quant_multiplier};
    use atlas_core::config::{
        ModelConfig, ModelOptQuantizationGroup, ModelOptWeightFormat, QuantizationConfig,
    };

    fn config_with(qc: QuantizationConfig) -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.quantization_config = Some(qc);
        config
    }

    fn nvfp4_group(name: &str) -> ModelOptQuantizationGroup {
        ModelOptQuantizationGroup {
            name: name.to_string(),
            weight_format: ModelOptWeightFormat::Nvfp4,
            group_size: 16,
            targets: vec![name.to_string()],
        }
    }

    /// `quant_algo = "MIXED_PRECISION"` where every `quantized_layers` entry
    /// (parsed into `config_groups` by
    /// `atlas_core::config::parsers::quantization::parse_modelopt_quantized_layers`)
    /// is plain NVFP4 must get the same multiplier as a checkpoint that labels
    /// itself "NVFP4" directly — same on-disk tensor format either way.
    #[test]
    fn mixed_precision_with_every_group_nvfp4_gets_the_nvfp4_multiplier() {
        let qc = QuantizationConfig {
            quant_method: "modelopt".to_string(),
            quant_algo: "MIXED_PRECISION".to_string(),
            format: String::new(),
            ignore_modules: vec!["lm_head".to_string()],
            config_groups: (0..48)
                .map(|i| nvfp4_group(&format!("model.language_model.layers.{i}.mlp.experts")))
                .collect(),
        };
        assert_eq!(
            quant_multiplier(&config_with(qc)),
            Some(NVFP4_PER_TENSOR_ALLOC_MULTIPLIER)
        );
    }

    /// A genuinely mixed checkpoint (one group not NVFP4) must keep
    /// falling through to the 1.3x heuristic — this fix only collapses
    /// the *uniformly*-NVFP4 case, not every MIXED_PRECISION checkpoint.
    #[test]
    fn mixed_precision_with_a_non_nvfp4_group_keeps_the_heuristic() {
        let qc = QuantizationConfig {
            quant_method: "modelopt".to_string(),
            quant_algo: "MIXED_PRECISION".to_string(),
            format: String::new(),
            ignore_modules: vec![],
            config_groups: vec![
                nvfp4_group("model.language_model.layers.0.mlp.experts"),
                ModelOptQuantizationGroup {
                    name: "model.language_model.layers.1.mlp.experts".to_string(),
                    weight_format: ModelOptWeightFormat::Fp8,
                    group_size: 0,
                    targets: vec!["model.language_model.layers.1.mlp.experts".to_string()],
                },
            ],
        };
        assert_eq!(quant_multiplier(&config_with(qc)), None);
    }

    /// `quant_algo = "MIXED_PRECISION"` with no typed groups at all (the
    /// ambiguous aggregate-only sidecar `parse_modelopt_mixed_groups`
    /// already declines to resolve) must not be treated as NVFP4 either.
    #[test]
    fn mixed_precision_with_no_config_groups_keeps_the_heuristic() {
        let qc = QuantizationConfig {
            quant_method: "modelopt".to_string(),
            quant_algo: "MIXED_PRECISION".to_string(),
            format: String::new(),
            ignore_modules: vec![],
            config_groups: vec![],
        };
        assert_eq!(quant_multiplier(&config_with(qc)), None);
    }

    /// Unaffected baseline: plain `quant_algo = "NVFP4"` gets the measured
    /// per-tensor-allocation multiplier.
    #[test]
    fn plain_nvfp4_is_unaffected() {
        let qc = QuantizationConfig {
            quant_method: "modelopt".to_string(),
            quant_algo: "NVFP4".to_string(),
            format: String::new(),
            ignore_modules: vec![],
            config_groups: vec![],
        };
        assert_eq!(
            quant_multiplier(&config_with(qc)),
            Some(NVFP4_PER_TENSOR_ALLOC_MULTIPLIER)
        );
    }

    /// The multiplier is the r8 receipt's resident/requested ratio, not the
    /// old mmap assumption: 90,290,393,088 / 77,531,021,560 = 1.1646.
    #[test]
    fn nvfp4_multiplier_matches_the_r8_allocation_audit() {
        let measured = 90_290_393_088f64 / 77_531_021_560f64;
        assert!((NVFP4_PER_TENSOR_ALLOC_MULTIPLIER - measured).abs() < 0.001);
        const { assert!(NVFP4_PER_TENSOR_ALLOC_MULTIPLIER > 1.02) };
    }
}

/// Per-model "never upload" filter for the serving store. DeepSeek-V4.1's engram tables
/// (~190 GB) are gathered from NVMe and must not be loaded — see
/// `spark_model::weight_loader::deepseek_v41::skip_tensor_for_serving`.
/// `ATLAS_QWEN4_TEXT_ONLY=1` (default off): serve Flash-Next text-only. The
/// vision tower's weights (~856 MiB BF16 on the QAD checkpoints) are not
/// loaded and image requests are refused like on a model without a vision
/// encoder. The model config keeps its vision block, so text prompts take
/// exactly the same path (prefix-cache image-pad checks included).
pub(crate) const QWEN4_TEXT_ONLY_SELECTOR: &str = "ATLAS_QWEN4_TEXT_ONLY";

/// Exact parse: unset or `0` is off, `1` is on, anything else is refused.
fn parse_qwen4_text_only(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => anyhow::bail!("{QWEN4_TEXT_ONLY_SELECTOR} must be 0 or 1, got {other:?}"),
    }
}

/// Whether this Flash-Next launch serves text only (always false elsewhere).
pub(crate) fn qwen4_text_only(config: &ModelConfig) -> Result<bool> {
    if !config.is_qwen4_exp() {
        return Ok(false);
    }
    parse_qwen4_text_only(std::env::var(QWEN4_TEXT_ONLY_SELECTOR).ok().as_deref())
}

/// The vision tower's tensors, under the flat or the nested checkpoint prefix.
fn is_vision_tower_tensor(name: &str) -> bool {
    name.starts_with("model.visual.") || name.starts_with("model.language_model.visual.")
}

/// Under pipeline parallelism (`pp_world_size > 1`), skip transformer layer
/// tensors outside this stage's `[pp_start, pp_end)` global layer slice and
/// skip `mtp.*` draft weights on non-head stages (`pp_rank > 0`). Shared
/// non-layer tensors (`embed_tokens`, `lm_head`, `norm`, `final_mixer`, etc.)
/// remain loaded on every stage so preflight and head/tail projections work.
fn is_pp_nonlocal_tensor(name: &str, pp_start: usize, pp_end: usize, pp_rank: usize) -> bool {
    if name.starts_with("mtp.") {
        return pp_rank > 0;
    }
    let tail = if let Some(pos) = name.find(".layers.") {
        &name[pos + ".layers.".len()..]
    } else if let Some(rest) = name.strip_prefix("layers.") {
        rest
    } else {
        return false;
    };
    let Some(end) = tail.find('.') else {
        return false;
    };
    let Ok(layer_idx) = tail[..end].parse::<usize>() else {
        return false;
    };
    layer_idx < pp_start || layer_idx >= pp_end
}

/// Under expert parallelism (`ep_world_size > 1`), skip routed expert tensors
/// outside this rank's `[ep_start, ep_end)` range (never skipping `mtp.*` or
/// `shared_expert`). Saves `(ep_world_size - 1) / ep_world_size` of disk read
/// and weight RAM at load time (~48 GiB/Spark on 4x EP=4).
pub(crate) fn is_ep_nonlocal_expert_tensor(name: &str, ep_start: usize, ep_end: usize) -> bool {
    if name.starts_with("mtp.") {
        return false;
    }
    let Some(pos) = name.find(".experts.") else {
        return false;
    };
    let tail = &name[pos + ".experts.".len()..];
    let Some(end) = tail.find('.') else {
        return false;
    };
    let Ok(expert_idx) = tail[..end].parse::<usize>() else {
        return false;
    };
    expert_idx < ep_start || expert_idx >= ep_end
}

fn serving_skip(
    config: &ModelConfig,
    text_only: bool,
) -> Option<spark_runtime::weights::TensorSkipFn> {
    let pp_skip = (config.pp_world_size > 1).then(|| {
        let (pp_start, pp_end) = config.pp_layer_range();
        (pp_start, pp_end, config.pp_rank)
    });
    let ep_skip =
        (config.ep_world_size > 1 && config.num_experts > 0).then(|| config.local_expert_range());
    if config.model_type == "deepseek_v41" {
        if pp_skip.is_some() || ep_skip.is_some() {
            return Some(std::sync::Arc::new(move |name: &str| {
                spark_model::weight_loader::deepseek_v41::skip_tensor_for_serving(name)
                    || pp_skip.is_some_and(|(pp_start, pp_end, pp_rank)| {
                        is_pp_nonlocal_tensor(name, pp_start, pp_end, pp_rank)
                    })
                    || ep_skip.is_some_and(|(ep_start, ep_end)| {
                        is_ep_nonlocal_expert_tensor(name, ep_start, ep_end)
                    })
            }));
        }
        return Some(std::sync::Arc::new(
            spark_model::weight_loader::deepseek_v41::skip_tensor_for_serving,
        ));
    }
    let ple_table_prefix = if config.is_qwen4_exp() {
        config.ple_offload_manifest.as_ref().map(|_| {
            format!(
                "{}.ple.ple_embedding.ngram_embedding",
                config.layer_prefix(1)
            )
        })
    } else {
        None
    };
    let qwen4_text_only = config.is_qwen4_exp() && text_only;
    if ple_table_prefix.is_none() && !qwen4_text_only && pp_skip.is_none() && ep_skip.is_none() {
        return None;
    }
    Some(std::sync::Arc::new(move |name: &str| {
        pp_skip.is_some_and(|(pp_start, pp_end, pp_rank)| {
            is_pp_nonlocal_tensor(name, pp_start, pp_end, pp_rank)
        }) || ep_skip
            .is_some_and(|(ep_start, ep_end)| is_ep_nonlocal_expert_tensor(name, ep_start, ep_end))
            || ple_table_prefix.as_deref().is_some_and(|table_prefix| {
                spark_model::layers::qwen4_ple::skip_offloaded_ple_tensor(table_prefix, name)
            })
            || (qwen4_text_only && is_vision_tower_tensor(name))
    }))
}

/// NVFP4 scale group size used by `QuantizedWeight`
/// (`spark-model/src/weight_map/quantized.rs`).
const NVFP4_GROUP: usize = 16;

/// Bytes an NVFP4 `QuantizedWeight` of logical shape `[n, k]` occupies:
/// packed nibbles `[n, k/2]` + per-group FP8 scales `[n, k/NVFP4_GROUP]`.
fn nvfp4_bytes(n: usize, k: usize) -> usize {
    n * k / 2 + n * k / NVFP4_GROUP
}

/// Bytes a `QuantizedWeight::transpose_for_gemm(_, n, k)` result occupies.
/// The transposed layout pads N up to 64 for cp.async alignment — see
/// `spark-model/src/weight_map/quantized.rs:59`.
fn nvfp4_transposed_bytes(n: usize, k: usize) -> usize {
    let n_pad = n.div_ceil(64) * 64;
    (k / 2) * n_pad + (k / NVFP4_GROUP) * n_pad
}

/// A representative SSM-layer module path for the ignore-list check
/// below. Layer index doesn't matter: `construction_overhead_bytes`
/// already assumes identical quantization treatment on every SSM layer
/// (one `per_ssm_layer` byte count computed once and multiplied by
/// `ssm_layers`), so layer 0 stands in for all of them.
const REPRESENTATIVE_LINEAR_ATTN_OUT_PROJ: &str = "model.layers.0.linear_attn.out_proj";

/// Does one `quantization_config.ignore`-style entry cover
/// `linear_attn.out_proj`?
///
/// Two shapes show up in the wild (`atlas-core/src/config/parsers/
/// quantization.rs` copies the raw JSON array verbatim — neither shape is
/// normalized before it reaches here):
///   * Already expanded to literal per-tensor module paths — substring
///     match, as this check has always done.
///   * An unexpanded glob straight from the checkpoint's `ignore` /
///     `exclude_modules` / `modules_to_not_convert` array (e.g. ModelOpt's
///     `*.linear_attn.*`, which is exactly what
///     `Qwen3.8-Flash-Next-NVFP4-Offload/config.json` ships) — matched
///     with the same `*`-as-wildcard semantics as the real per-tensor
///     dispatch, `spark_model::quant_format::module_matches_pattern`
///     (duplicated here, not imported, to keep this estimator's change
///     self-contained and because `quant_format` is `pub(crate)` to that
///     crate). A glob is detected by the presence of `*`; without one,
///     this falls back to the original substring check so no
///     already-correct call site changes behavior.
fn ignore_pattern_covers_out_proj(pattern: &str) -> bool {
    const NEEDLE: &str = "linear_attn.out_proj";
    if !pattern.contains('*') {
        return pattern.contains(NEEDLE);
    }
    let path = REPRESENTATIVE_LINEAR_ATTN_OUT_PROJ;
    let segments: Vec<&str> = pattern.split('*').collect();
    let mut rest = path;
    let first = segments[0];
    if !first.is_empty() {
        let Some(stripped) = rest.strip_prefix(first) else {
            return false;
        };
        rest = stripped;
    }
    for seg in &segments[1..segments.len() - 1] {
        if seg.is_empty() {
            continue;
        }
        match rest.find(seg) {
            Some(pos) => rest = &rest[pos + seg.len()..],
            None => return false,
        }
    }
    let last = segments[segments.len() - 1];
    last.is_empty() || rest.ends_with(last)
}

#[cfg(test)]
mod ignore_pattern_tests {
    use super::ignore_pattern_covers_out_proj;

    #[test]
    fn unexpanded_glob_covers_out_proj() {
        // Qwen3.8-Flash-Next-NVFP4-Offload's actual ignore entry.
        assert!(ignore_pattern_covers_out_proj("*.linear_attn.*"));
    }

    #[test]
    fn glob_anchored_on_out_proj_suffix_matches() {
        assert!(ignore_pattern_covers_out_proj("*.linear_attn.out_proj"));
    }

    #[test]
    fn glob_for_an_unrelated_module_does_not_match() {
        assert!(!ignore_pattern_covers_out_proj("*.mlp.gate_proj.*"));
        assert!(!ignore_pattern_covers_out_proj("lm_head"));
    }

    #[test]
    fn already_expanded_literal_path_still_matches_by_substring() {
        // Pre-existing behavior for checkpoints whose ignore list is
        // already per-tensor literal paths, not globs.
        assert!(ignore_pattern_covers_out_proj(
            "model.layers.17.linear_attn.out_proj"
        ));
    }

    #[test]
    fn literal_path_for_a_different_module_does_not_match() {
        assert!(!ignore_pattern_covers_out_proj(
            "model.layers.17.linear_attn.in_proj_qkv"
        ));
    }
}

/// GPU bytes the model *builder* retains on top of the weight store,
/// broken down by source. Absolute bytes, not a ratio.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ConstructionOverhead {
    /// Per-linear-attention-layer buffers built by the SSM loader arms
    /// (`weight_loader/qwen35_dense.rs` and
    /// `weight_loader/qwen35/load_layers/linear_attn_arms.rs`).
    pub(crate) ssm: usize,
    /// Transposed NVFP4 copies of the dense FFN projections
    /// (`ATLAS_FFN_M16_TRANSPOSED`, default on).
    pub(crate) ffn_transposed: usize,
    /// Transposed NVFP4 q/k/v/o copies on full-attention layers.
    pub(crate) attn_transposed: usize,
    /// FlashInfer/CUTLASS 128x4 copies of gate/up/down FP8 block scales.
    /// Packed checkpoint weights are reused in place; only scale bytes and
    /// three alpha FP32 scalars are retained per dense layer.
    pub(crate) ffn_flashinfer_scales: usize,
    /// Additional retained merged gate/up operands. This is the packed-weight
    /// delta relative to the former separate gate/up FlashInfer preparation;
    /// its physical scales and shared alpha replace bytes already counted in
    /// `ffn_flashinfer_scales`.
    pub(crate) ffn_flashinfer_merged_gate_up: usize,
    /// Per-layer activation arena retained after the first maximum-size
    /// FlashInfer FFN prefill: packed E2M1 values, padded 128x4 E4M3 scales,
    /// and one FP32 scalar at exact M8192/K=intermediate.
    pub(crate) ffn_flashinfer_scratch: usize,
}

impl ConstructionOverhead {
    pub(crate) fn total(&self) -> usize {
        self.ssm
            + self.ffn_transposed
            + self.attn_transposed
            + self.ffn_flashinfer_scales
            + self.ffn_flashinfer_merged_gate_up
            + self.ffn_flashinfer_scratch
    }
}

fn flashinfer_merged_gate_up_overhead_bytes(num_hidden_layers: usize) -> usize {
    spark_model::weight_map::flashinfer_ffn_admission::qwen38_merged_gate_up_additional_bytes()
        * num_hidden_layers
}

#[cfg(test)]
mod construction_overhead_tests {
    use super::{ConstructionOverhead, flashinfer_merged_gate_up_overhead_bytes};

    #[test]
    fn merged_gate_up_charge_uses_the_admission_ssot() {
        assert_eq!(flashinfer_merged_gate_up_overhead_bytes(1), 89_128_956);
        assert_eq!(flashinfer_merged_gate_up_overhead_bytes(64), 5_704_253_184);
        let overhead = ConstructionOverhead {
            ffn_flashinfer_merged_gate_up: flashinfer_merged_gate_up_overhead_bytes(64),
            ..ConstructionOverhead::default()
        };
        assert_eq!(overhead.total(), 5_704_253_184);
    }
}

#[cfg(test)]
mod memlean_overhead_tests {
    use super::{OverheadSelectors, construction_overhead_bytes_with};
    use atlas_core::config::{LayerType, ModelConfig, QuantizationConfig};

    /// Qwen3.8-27B-NVFP4: 64 layers (16 attention, 48 SSM), dense FFN, FP8
    /// linear-attention projections, `mtp*` the only ignored modules.
    fn qwen38_27b() -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen3_5".into();
        config.hidden_size = 5120;
        config.num_hidden_layers = 64;
        config.intermediate_size = 17408;
        config.num_experts = 0;
        config.linear_num_key_heads = 16;
        config.linear_key_head_dim = 128;
        config.linear_num_value_heads = 48;
        config.linear_value_head_dim = 128;
        config.num_attention_heads = 24;
        config.num_key_value_heads = 4;
        config.head_dim = 256;
        config.attn_gated = true;
        config.layer_types = (0..64)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();
        config.quantization_config = Some(QuantizationConfig {
            quant_method: "modelopt".into(),
            quant_algo: "MIXED_PRECISION".into(),
            format: String::new(),
            ignore_modules: vec!["mtp*".into(), "mtp.layers.0*".into()],
            config_groups: Vec::new(),
        });
        config
    }

    /// `flux-qwen38-27b-dflash2.env`: transposed FFN copies + FlashInfer FFN.
    fn recipe() -> OverheadSelectors {
        OverheadSelectors {
            ffn_m16_transposed: true,
            ffn_flashinfer: true,
            ..OverheadSelectors::default()
        }
    }

    const BF16_QKVZ_PER_LAYER: usize = 16_384 * 5_120 * 2;
    const MERGED_GATE_UP_64_LAYERS: usize = 5_704_253_184;
    const FFN_SCRATCH_PER_LAYER: usize = 80_216_068;

    #[test]
    fn qwen38_27b_recipe_estimate_matches_the_rc8_log() {
        let overhead = construction_overhead_bytes_with(&qwen38_27b(), &recipe());
        assert_eq!(overhead.ssm, 18_858_639_360);
        assert_eq!(overhead.ffn_transposed, 9_625_927_680);
        assert_eq!(overhead.attn_transposed, 943_718_400);
        assert_eq!(overhead.ffn_flashinfer_scales, 1_069_548_288);
        assert_eq!(
            overhead.ffn_flashinfer_merged_gate_up,
            MERGED_GATE_UP_64_LAYERS
        );
        assert_eq!(overhead.ffn_flashinfer_scratch, 64 * FFN_SCRATCH_PER_LAYER);
        // Logged as "38.50 GB" (GiB) by the UK R5 q27-rc8 arm.
        assert_eq!(overhead.total(), 41_335_915_264);
    }

    #[test]
    fn dense_bf16_qkvz_release_drops_one_concat_per_ssm_layer() {
        let base = construction_overhead_bytes_with(&qwen38_27b(), &recipe());
        let lean = construction_overhead_bytes_with(
            &qwen38_27b(),
            &OverheadSelectors {
                dense_release_bf16_qkvz: true,
                ..recipe()
            },
        );
        assert_eq!(base.ssm - lean.ssm, 48 * BF16_QKVZ_PER_LAYER);
        assert_eq!(base.total() - lean.total(), 8_053_063_680);
    }

    #[test]
    fn each_loader_honours_only_its_own_qkvz_selector() {
        let dense_base = construction_overhead_bytes_with(&qwen38_27b(), &recipe());
        let dense_with_qwen4_selector = construction_overhead_bytes_with(
            &qwen38_27b(),
            &OverheadSelectors {
                qwen4_release_bf16_qkvz: true,
                ..recipe()
            },
        );
        assert_eq!(dense_with_qwen4_selector.ssm, dense_base.ssm);

        let moe = ModelConfig::qwen3_next_80b_nvfp4();
        let moe_base = construction_overhead_bytes_with(&moe, &OverheadSelectors::default());
        let moe_with_dense_selector = construction_overhead_bytes_with(
            &moe,
            &OverheadSelectors {
                dense_release_bf16_qkvz: true,
                ..OverheadSelectors::default()
            },
        );
        assert_eq!(moe_with_dense_selector.ssm, moe_base.ssm);
        let moe_with_qwen4_selector = construction_overhead_bytes_with(
            &moe,
            &OverheadSelectors {
                qwen4_release_bf16_qkvz: true,
                ..OverheadSelectors::default()
            },
        );
        assert_eq!(
            moe_base.ssm - moe_with_qwen4_selector.ssm,
            moe.num_ssm_layers() * moe.ssm_qkvz_size() * moe.hidden_size * 2
        );
    }

    #[test]
    fn forced_bf16_gdn_weights_keep_the_concat() {
        let base = construction_overhead_bytes_with(&qwen38_27b(), &recipe());
        let forced = construction_overhead_bytes_with(
            &qwen38_27b(),
            &OverheadSelectors {
                gdn_bf16_forced: true,
                dense_release_bf16_qkvz: true,
                ..recipe()
            },
        );
        assert_eq!(forced.ssm, base.ssm);
    }

    #[test]
    fn aliased_gate_up_retains_no_merged_copy() {
        let lean = construction_overhead_bytes_with(
            &qwen38_27b(),
            &OverheadSelectors {
                ffn_gate_up_alias: true,
                ..recipe()
            },
        );
        assert_eq!(lean.ffn_flashinfer_merged_gate_up, 0);
        // The physical 128x4 scale copies stay: FlashInfer reads them.
        assert_eq!(lean.ffn_flashinfer_scales, 1_069_548_288);
    }

    #[test]
    fn shared_ffn_scratch_is_charged_once() {
        let lean = construction_overhead_bytes_with(
            &qwen38_27b(),
            &OverheadSelectors {
                shared_scratch: true,
                ..recipe()
            },
        );
        assert_eq!(lean.ffn_flashinfer_scratch, FFN_SCRATCH_PER_LAYER);
    }

    #[test]
    fn memlean_selectors_compose() {
        let lean = construction_overhead_bytes_with(
            &qwen38_27b(),
            &OverheadSelectors {
                dense_release_bf16_qkvz: true,
                ffn_gate_up_alias: true,
                shared_scratch: true,
                ..recipe()
            },
        );
        assert_eq!(
            lean.total(),
            41_335_915_264
                - 48 * BF16_QKVZ_PER_LAYER
                - MERGED_GATE_UP_64_LAYERS
                - 63 * FFN_SCRATCH_PER_LAYER
        );
        assert_eq!(lean.total(), 22_524_986_116);
    }

    #[test]
    fn memlean_selectors_change_nothing_without_flashinfer() {
        let no_flashinfer = OverheadSelectors {
            ffn_m16_transposed: true,
            ..OverheadSelectors::default()
        };
        let base = construction_overhead_bytes_with(&qwen38_27b(), &no_flashinfer);
        let lean = construction_overhead_bytes_with(
            &qwen38_27b(),
            &OverheadSelectors {
                ffn_gate_up_alias: true,
                shared_scratch: true,
                ..no_flashinfer
            },
        );
        assert_eq!(lean.total(), base.total());
    }
}

/// Estimate the GPU memory the model builder allocates *and keeps* after
/// the weight store finishes loading.
///
/// Why this exists: the load pre-flight is the only guard against
/// over-allocating at startup, and on GB10's unified memory an
/// over-allocation is a host OOM, not a clean CUDA failure. Before this
/// function the pre-flight modelled construction as a flat 1.02× of the
/// on-disk bytes, on the belief that the SSM QKV/Z concat was freed once
/// the layer had been quantized. By default it is not: `qkvz_dense` is
/// moved into `SsmWeights.in_proj_qkvz` (`qwen35_dense.rs:378`,
/// `linear_attn_arms.rs:241`) and lives for the process lifetime,
/// alongside *both* NVFP4 copies. Neither `DenseWeight` nor
/// `QuantizedWeight` implements `Drop`, and the loader deliberately does
/// not call `gpu.free()` (BUG #29: freeing on GB10 UVM posts in-band TLB
/// invalidations that corrupt neighbouring allocations). So every buffer
/// counted here is retained, by design — *unless* the loader's own
/// opt-in release flags say otherwise (`ATLAS_QWEN4_RELEASE_BF16_QKVZ=1`
/// frees the QKVZ concat during that same quiescent construction window,
/// once it's safe; this function mirrors that condition instead of assuming
/// worst-case — see `qkvz_bf16_released` below).
///
/// Scope: only hybrid linear-attention (GDN/SSM) checkpoints. Models with
/// no `linear_attention` layers return zero and keep their previous
/// pre-flight behaviour exactly. MoE expert transposition is excluded on
/// purpose — it has its own free-memory guard at
/// `weight_loader/qwen35/load_layers.rs:102` and skips itself when it
/// would not fit, so charging it here would false-OOM MoE checkpoints.
///
/// Not modelled (all post-store, all comparatively small): the `lm_head`
/// NVFP4 quantization, the MTP head, ViT scratch, the DFlash drafter, and
/// the split-K FP32 workspaces. Together ~2 GB on Qwen3.8-27B.
pub(crate) fn construction_overhead_bytes(config: &ModelConfig) -> ConstructionOverhead {
    construction_overhead_bytes_with(config, &OverheadSelectors::from_env())
}

/// The environment switches `construction_overhead_bytes` mirrors, read once
/// so the estimate and its tests share one pure function.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct OverheadSelectors {
    /// `ATLAS_GDN_BF16_WEIGHTS=1`: the BF16 fallback path keeps every BF16 copy.
    pub(crate) gdn_bf16_forced: bool,
    /// `ATLAS_QWEN4_RELEASE_BF16_QKVZ=1`, honoured by the MoE hybrid loader.
    pub(crate) qwen4_release_bf16_qkvz: bool,
    /// `ATLAS_MEMLEAN_SSM_BF16_QKVZ=1`, honoured by the dense hybrid loader.
    pub(crate) dense_release_bf16_qkvz: bool,
    /// `ATLAS_FFN_M16_TRANSPOSED` (default on).
    pub(crate) ffn_m16_transposed: bool,
    /// `ATLAS_PREFILL_FFN_FLASHINFER=1`.
    pub(crate) ffn_flashinfer: bool,
    /// `ATLAS_MEMLEAN_FFN_GATE_UP_ALIAS=1`: the merged gate/up operand replaces
    /// the separate gate/up packed weights instead of adding to them.
    pub(crate) ffn_gate_up_alias: bool,
    /// `ATLAS_MEMLEAN_SHARED_SCRATCH=1`: one activation scratch for every
    /// dense FFN layer.
    pub(crate) shared_scratch: bool,
}

impl OverheadSelectors {
    fn from_env() -> Self {
        let exact_one = |name: &str| std::env::var(name).ok().as_deref() == Some("1");
        // A malformed MEMLEAN value fails the loader itself; the estimate
        // then charges the conservative (unreleased) bytes.
        let memlean = |name: &str| spark_model::memlean::selected(name).unwrap_or(false);
        Self {
            gdn_bf16_forced: exact_one("ATLAS_GDN_BF16_WEIGHTS"),
            qwen4_release_bf16_qkvz: exact_one("ATLAS_QWEN4_RELEASE_BF16_QKVZ"),
            dense_release_bf16_qkvz: memlean(spark_model::memlean::SSM_BF16_QKVZ),
            ffn_m16_transposed: spark_model::layers::ffn_m16_transposed_enabled(),
            ffn_flashinfer: spark_model::layers::prefill_ffn_flashinfer_enabled().unwrap_or(false),
            ffn_gate_up_alias: memlean(spark_model::memlean::FFN_GATE_UP_ALIAS),
            shared_scratch: memlean(spark_model::memlean::SHARED_SCRATCH),
        }
    }
}

pub(crate) fn construction_overhead_bytes_with(
    config: &ModelConfig,
    selectors: &OverheadSelectors,
) -> ConstructionOverhead {
    let ssm_layers = config.num_ssm_layers();
    if ssm_layers == 0 {
        return ConstructionOverhead::default();
    }

    let h = config.hidden_size;
    let qkvz = config.ssm_qkvz_size();
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;

    // `out_proj` is dequantized to a fresh BF16 buffer by
    // `load_ssm_qwen35` only when it is packed on disk. Checkpoints that
    // list it in the quantizer's ignore list ship it as BF16 and the
    // loader aliases the store pointer instead — no extra allocation.
    //
    // The ignore list can arrive two ways: already expanded to literal
    // per-tensor module paths (substring-matchable, the only case this
    // check used to handle), or as an unexpanded glob straight from
    // `quantization_config.ignore`/`exclude_modules`/`modules_to_not_convert`
    // (`atlas-core/src/config/parsers/quantization.rs` copies those arrays
    // verbatim — no glob expansion happens before they land here). A glob
    // such as `*.linear_attn.*` (NVFP4-Offload's own ignore list) covers
    // `out_proj` but does not literally spell out "linear_attn.out_proj"
    // as a contiguous substring, so a plain `.contains()` check would miss it
    // and charge a fresh dequant buffer that `load_ssm_qwen35` never allocates
    // (it takes the alias branch), overestimating ~1.03 GiB across 36 SSM
    // layers.
    let out_proj_packed_on_disk = config.quantization_config.as_ref().is_none_or(|qc| {
        !qc.ignore_modules
            .iter()
            .any(|m| ignore_pattern_covers_out_proj(m))
    });
    let out_proj_bf16_dequant = if out_proj_packed_on_disk {
        h * value_dim * 2
    } else {
        0
    };

    // `ATLAS_QWEN4_RELEASE_BF16_QKVZ=1` (default off; see
    // `weight_loader/qwen35/load_layers/linear_attn_arms.rs`) frees the BF16
    // QKVZ concat during the quiescent per-layer construction window once its
    // NVFP4 copy exists (`gpu.synchronize` then `gpu.free`, matching
    // `release_owned_dequant_source`), saving ~80 MiB/layer (~2.81 GiB across
    // 36 SSM layers). `ATLAS_GDN_BF16_WEIGHTS=1` forces the BF16 fallback path
    // and disables the release (matching the loader's `retain_out_proj` check),
    // mirrored here via the same two env reads.
    // The dense hybrid loader (`num_experts == 0`, `qwen35_dense.rs`)
    // ignores that qwen4 selector and honours `ATLAS_MEMLEAN_SSM_BF16_QKVZ=1`
    // instead, with the same `ATLAS_GDN_BF16_WEIGHTS` exclusion.
    let qkvz_bf16_released = !selectors.gdn_bf16_forced
        && if config.num_experts == 0 {
            selectors.dense_release_bf16_qkvz
        } else {
            selectors.qwen4_release_bf16_qkvz
        };
    let qkvz_bf16_concat = if qkvz_bf16_released { 0 } else { qkvz * h * 2 };

    let per_ssm_layer =
        // qkvz BF16 concat, retained as `SsmWeights.in_proj_qkvz` unless
        // `ATLAS_QWEN4_RELEASE_BF16_QKVZ=1` already freed it (see above).
        qkvz_bf16_concat
        // qkvz NVFP4 copy + its transposed duplicate.
        + nvfp4_bytes(qkvz, h)
        + nvfp4_transposed_bytes(qkvz, h)
        // BF16 dequant of a packed on-disk out_proj, never freed.
        + out_proj_bf16_dequant
        // out_proj NVFP4 + transposed duplicate + FP8 prefill predequant
        // (`layers/qwen3_ssm/init.rs::predequant_for_prefill`).
        + nvfp4_bytes(h, value_dim)
        + nvfp4_transposed_bytes(h, value_dim)
        + h * value_dim
        // interleave_ba BF16 output.
        + config.ssm_ba_size() * h * 2;

    // Dense-FFN transposed NVFP4 copies. MoE checkpoints take the
    // expert path instead (guarded separately, see above).
    let ffn_transposed = if config.num_experts == 0 && selectors.ffn_m16_transposed {
        let inter = config.intermediate_size;
        let per_layer = nvfp4_transposed_bytes(inter, h)   // gate
                + nvfp4_transposed_bytes(inter, h)             // up
                + nvfp4_transposed_bytes(h, inter); // down
        per_layer * config.num_hidden_layers
    } else {
        0
    };

    // Transposed q/k/v/o on full-attention layers. Unconditional on both
    // the dense (`qwen35_dense.rs:293`) and MoE
    // (`load_layers/attention_arms.rs:196`) hybrid paths.
    let attn_transposed = {
        let head_dim = config.head_dim;
        let q_out = config.num_attention_heads * head_dim * if config.attn_gated { 2 } else { 1 };
        let kv_out = config.num_key_value_heads * head_dim;
        let per_layer = nvfp4_transposed_bytes(q_out, h)
            + 2 * nvfp4_transposed_bytes(kv_out, h)
            + nvfp4_transposed_bytes(h, config.num_attention_heads * head_dim);
        per_layer * config.num_attention_layers()
    };

    let (ffn_flashinfer_scales, ffn_flashinfer_merged_gate_up, ffn_flashinfer_scratch) =
        if config.num_experts == 0 && selectors.ffn_flashinfer {
            let inter = config.intermediate_size;
            let per_layer = inter * h / NVFP4_GROUP // gate physical scales
                + inter * h / NVFP4_GROUP // up physical scales
                + h * inter / NVFP4_GROUP // down physical scales
                + 3 * std::mem::size_of::<f32>(); // three alpha scalars
            let max_m = 8_192usize;
            let scratch_per_layer = max_m * inter / 2 // packed E2M1 activation
                + max_m * inter / NVFP4_GROUP // padded 128x4 scales; M already 128-aligned
                + std::mem::size_of::<f32>(); // legacy absmax allocation retained by the arena
            // Aliased gate/up views replace the separate packed weights the
            // store held, so the merged operand adds nothing on top of it.
            let merged_gate_up = if selectors.ffn_gate_up_alias {
                0
            } else {
                flashinfer_merged_gate_up_overhead_bytes(config.num_hidden_layers)
            };
            let scratch_copies = if selectors.shared_scratch {
                1
            } else {
                config.num_hidden_layers
            };
            (
                per_layer * config.num_hidden_layers,
                merged_gate_up,
                scratch_per_layer * scratch_copies,
            )
        } else {
            (0, 0, 0)
        };

    ConstructionOverhead {
        ssm: per_ssm_layer * ssm_layers,
        ffn_transposed,
        attn_transposed,
        ffn_flashinfer_scales,
        ffn_flashinfer_merged_gate_up,
        ffn_flashinfer_scratch,
    }
}

pub(crate) fn load_weight_store(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    model_dir: &Path,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    ep_rank: usize,
    ep_size: usize,
    oom_reserve_bytes: usize,
) -> Result<spark_runtime::weights::WeightStore> {
    use spark_runtime::weights::WeightLoader;
    let text_only = qwen4_text_only(config)?;
    if text_only {
        tracing::info!(
            "QWEN4_TEXT_ONLY_ENGAGED selector={QWEN4_TEXT_ONLY_SELECTOR}: the vision tower is not \
             loaded; image requests are refused"
        );
    }
    let mult = quant_multiplier(config);
    let overhead = construction_overhead_bytes(config);
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    if overhead.total() > 0 {
        // The window between "weight store loaded" and "KV cache sized"
        // used to consume tens of GB with no byte-count log line at all,
        // which is how the 1.02× under-estimate stayed hidden. Print the
        // prediction here so it can be diffed against the free-memory
        // deltas either side of model construction without an audit.
        tracing::info!(
            "Model-construction overhead estimate: {:.2} GB retained after the weight \
             store loads — SSM linear-attn {:.2} GB ({} layers), dense-FFN transposes \
             {:.2} GB, FlashInfer FFN scales {:.2} GB, merged gate/up {:.2} GB, FlashInfer FFN activation scratch {:.2} GB, attention transposes {:.2} GB ({} layers). Charged to the load \
             pre-flight on top of the {}x on-disk ratio. Excludes lm_head / MTP / \
             ViT / DFlash drafter.",
            gib(overhead.total()),
            gib(overhead.ssm),
            config.num_ssm_layers(),
            gib(overhead.ffn_transposed),
            gib(overhead.ffn_flashinfer_scales),
            gib(overhead.ffn_flashinfer_merged_gate_up),
            gib(overhead.ffn_flashinfer_scratch),
            gib(overhead.attn_transposed),
            config.num_attention_layers(),
            mult.map_or_else(|| "1.30–1.50 (auto)".to_string(), |m| format!("{m:.2}")),
        );
    }
    let use_fast_load =
        !args.no_fast_load && std::env::var("ATLAS_FAST_LOAD").ok().as_deref() != Some("0");
    #[cfg(unix)]
    let grouped_routed_originals = {
        use spark_runtime::fast_weights::{ROUTED_GROUPS_ENV, routed_grouping_requested};
        let enabled = routed_grouping_requested(std::env::var_os(ROUTED_GROUPS_ENV).as_deref())?;
        if enabled {
            // Plain, native MTP (`--speculative`, in-checkpoint or
            // `--mtp-from-path`), and DFlash (`--dflash`, loaded from its own
            // separate drafter store after target layers and group release)
            // are admitted: groups hold only `model.language_model.layers.*`
            // routed originals (`routed_groups.rs` PREFIX); the MTP layer
            // loads under weight prefix `mtp` (`qwen35.rs` qwen4_mtp_config)
            // and DFlash loads from `dflash_args.drafter_store`, so neither
            // attaches a target group or reads target routed originals after
            // release (`factory/build.rs` transpose pass before drafter build).
            // Self- and n-gram speculation stay refused.
            anyhow::ensure!(
                use_fast_load
                    && config.is_qwen4_exp()
                    && config.num_hidden_layers == 48
                    && config.num_experts == 512
                    && config.hidden_size == 2560
                    && config.moe_intermediate_size == 640
                    && ep_rank == 0
                    && ep_size == 1
                    && config.ep_rank == 0
                    && config.ep_world_size == 1
                    && config.tp_world_size == 1
                    && config.tp_rank == 0
                    && !args.self_speculative
                    && !args.ngram_speculative,
                "{ROUTED_GROUPS_ENV}=1 requires canonical Qwen4 target geometry, fast loading, TP1/EP1 and Plain, native MTP, or DFlash (no self- or n-gram speculation)"
            );
        }
        enabled
    };
    // Strict sub-flag: unset/0/1 only, and 1 requires the grouping flag.
    #[cfg(unix)]
    let grouped_routed_originals_eager =
        spark_runtime::fast_weights::routed_grouping_eager_requested(
            std::env::var_os(spark_runtime::fast_weights::ROUTED_GROUPS_EAGER_ENV).as_deref(),
            grouped_routed_originals,
        )?;
    let store = if use_fast_load {
        #[cfg(unix)]
        {
            tracing::info!("Using fast weight loader (O_DIRECT + pipelined read/copy)");
            let mut loader = if ep_size > 1 {
                spark_runtime::fast_weights::FastSafetensorsLoader::with_ep(
                    ep_rank,
                    ep_size,
                    config.num_experts,
                )
            } else {
                spark_runtime::fast_weights::FastSafetensorsLoader::new()
            };
            loader.peak_memory_multiplier = mult;
            loader.construction_overhead_bytes = overhead.total();
            loader.extra_skip = serving_skip(config, text_only);
            loader.grouped_routed_originals = grouped_routed_originals;
            loader.grouped_routed_originals_eager = grouped_routed_originals_eager;
            loader
                .load(model_dir, gpu, oom_reserve_bytes)
                .context("Failed to load model weights (fast loader)")?
        }
        #[cfg(not(unix))]
        {
            anyhow::bail!(
                "Fast weight loader requires a Unix host (needs O_DIRECT / posix_fadvise; \
                 pass --no-fast-load or ATLAS_FAST_LOAD=0)"
            );
        }
    } else {
        let mut loader = if ep_size > 1 {
            spark_runtime::weights::SafetensorsLoader::with_ep(ep_rank, ep_size, config.num_experts)
        } else {
            spark_runtime::weights::SafetensorsLoader::new()
        };
        loader.peak_memory_multiplier = mult;
        loader.construction_overhead_bytes = overhead.total();
        loader.extra_skip = serving_skip(config, text_only);
        loader
            .load(model_dir, gpu, oom_reserve_bytes)
            .context("Failed to load model weights")?
    };
    tracing::info!("Loaded {} weight tensors", store.len());
    Ok(store)
}

pub(crate) fn auto_detect_weight_prefix(
    store: &spark_runtime::weights::WeightStore,
    config: &mut ModelConfig,
) {
    if config.weight_prefix.is_empty() && config.nested_config {
        config.weight_prefix = if store.contains("language_model.model.embed_tokens.weight") {
            "language_model.model".to_string()
        } else if store.contains("model.language_model.embed_tokens.weight") {
            "model.language_model".to_string()
        } else {
            let probe = format!(".layers.{}.", config.pp_layer_offset);
            let scanned = store
                .names()
                .find(|k| k.contains(&probe))
                .and_then(|k| k.split(&probe).next())
                .map(|s| s.to_string());
            if let Some(ref prefix) = scanned {
                tracing::info!("Auto-detected weight prefix: '{prefix}'");
            }
            scanned.unwrap_or_else(|| "model".to_string())
        };
    }
    if !config.weight_prefix.is_empty() {
        tracing::info!("Weight prefix: {}", config.weight_prefix);
    }
}

pub(crate) fn load_dflash_drafter(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    verify_mode: spark_model::weight_loader::DsparkVerifyMode,
    tc_parity_ar_reference_rows: Option<usize>,
) -> Result<
    Option<(
        spark_runtime::weights::WeightStore,
        spark_model::weight_loader::DflashConfig,
    )>,
> {
    use spark_runtime::weights::WeightLoader;
    if !args.dflash {
        let engaged = spark_model::layers::configure_decode_tc_parity(tc_parity_ar_reference_rows);
        if let Some(verify_rows) = tc_parity_ar_reference_rows {
            anyhow::ensure!(
                engaged,
                "ATLAS_DECODE_TC_PARITY_REFERENCE_ROWS={verify_rows} was admitted, but target TC parity did not engage"
            );
            tracing::warn!(
                schema = 1,
                verify_rows,
                scheduler_path = "ordinary-target-only",
                drafter_loaded = false,
                "QWEN38_TC_PARITY_AR_REFERENCE_ENGAGED"
            );
        }
        return Ok(None);
    }
    anyhow::ensure!(
        tc_parity_ar_reference_rows.is_none(),
        "the TC-parity AR reference cannot load a DFlash drafter"
    );
    let drafter_id = args
        .draft_model
        .clone()
        .or_else(|| ptx_set.dflash.as_ref().map(|d| d.draft_model.to_string()))
        .context(
            "--dflash set but no drafter HF id provided: pass --draft-model <ID> \
             or use a target whose MODEL.toml has a [dflash] section",
        )?;
    tracing::info!("DFlash: resolving drafter '{drafter_id}'");
    let drafter_dir =
        crate::model_resolver::resolve_model_dir(&drafter_id, args.cache_dir.as_deref())
            .context("Failed to resolve DFlash drafter checkpoint")?;
    let drafter_config_json = std::fs::read_to_string(drafter_dir.join("config.json"))
        .with_context(|| {
            format!(
                "Failed to read drafter config.json at {}",
                drafter_dir.display()
            )
        })?;
    let drafter_config =
        spark_model::weight_loader::dflash_loader::parse_dflash_config(&drafter_config_json)?;
    // Validate the requested planner before mapping the multi-GB drafter.
    // Dynamic modes must never degrade silently to static verify-all.
    drafter_config.validate_verify_mode(verify_mode)?;
    if let Some(manifest) =
        spark_model::weight_loader::dflash_loader::native_flash_next_finiteness_manifest(
            &drafter_config,
        )?
    {
        let receipt =
            spark_runtime::weights::attest_exact_bf16_safetensors(&drafter_dir, &manifest)
                .context("Native Flash-Next V3 pre-upload BF16 receipt failed")?;
        tracing::info!(
            file = %receipt.file_name,
            bytes = receipt.file_bytes,
            sha256 = %receipt.file_sha256,
            tensors = receipt.tensor_count,
            elements = receipt.element_count,
            "Native Flash-Next V3 BF16 finiteness receipt PASS"
        );
    }
    let mut loader = spark_runtime::weights::SafetensorsLoader::new();
    loader.peak_memory_multiplier = None;
    let drafter_store = loader
        .load(&drafter_dir, gpu, 0)
        .context("Failed to load DFlash drafter weights")?;
    tracing::info!(
        "DFlash drafter store: {} tensors, {} bytes",
        drafter_store.len(),
        drafter_store.total_bytes()
    );
    // Decode TC parity may only mirror a verify that runs the tensor-core
    // routes: record the configured verify width (γ + 1) at every model load,
    // before the first decode, so the gate fails closed when it cannot hold.
    spark_model::layers::configure_decode_tc_parity(Some(
        spark_model::layers::dflash_head::effective_verify_rows(
            drafter_config.resolve_draft_count(args.dflash_gamma)?,
        ),
    ));
    Ok(Some((drafter_store, drafter_config)))
}

/// Selectively load the dense-target tensors that an external DFlash
/// checkpoint omitted because they were shared during training.
pub(crate) fn load_dflash_donor(
    args: &cli::ServeArgs,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
) -> Result<Option<spark_runtime::weights::WeightStore>> {
    use spark_runtime::weights::WeightLoader;
    let Some(donor_id) = args.dflash_donor_model.as_ref() else {
        return Ok(None);
    };
    anyhow::ensure!(
        args.dflash,
        "--dflash-donor-model requires --dflash and an explicit --draft-model"
    );
    let donor_dir = crate::model_resolver::resolve_model_dir(donor_id, args.cache_dir.as_deref())
        .context("Failed to resolve DFlash donor checkpoint")?;
    let allow = [
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "lm_head.weight",
        "model.lm_head.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let mut loader = spark_runtime::weights::SafetensorsLoader::new();
    loader.peak_memory_multiplier = Some(1.0);
    loader.tensor_allowlist = Some(allow);
    let donor = loader
        .load(&donor_dir, gpu, 0)
        .context("Failed to selectively load DFlash donor embedding/lm_head")?;
    anyhow::ensure!(
        donor.len() == 2,
        "DFlash donor must expose exactly one embedding and one lm_head tensor; selectively loaded {} tensors: {:?}",
        donor.len(),
        donor.names().collect::<Vec<_>>(),
    );
    tracing::info!(
        "DFlash donor: selectively loaded {} tensors / {} bytes from {}",
        donor.len(),
        donor.total_bytes(),
        donor_dir.display(),
    );
    Ok(Some(donor))
}

#[cfg(test)]
mod text_only_tests {
    use super::{is_vision_tower_tensor, parse_qwen4_text_only, serving_skip};
    use atlas_core::config::ModelConfig;

    fn flash_next(ple_manifest: bool) -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".to_string();
        config.weight_prefix = "model.language_model".to_string();
        config.ple_offload_manifest = ple_manifest.then(|| "ple-offload/manifest.json".to_string());
        config
    }

    #[test]
    fn selector_is_exact_and_default_off() {
        assert!(!parse_qwen4_text_only(None).unwrap());
        assert!(!parse_qwen4_text_only(Some("0")).unwrap());
        assert!(parse_qwen4_text_only(Some("1")).unwrap());
        for bad in ["", "true", " 1", "yes"] {
            assert!(parse_qwen4_text_only(Some(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn text_only_skips_exactly_the_vision_tower() {
        let skip = serving_skip(&flash_next(false), true).expect("text-only installs a skip");
        for name in [
            "model.visual.patch_embed.proj.weight",
            "model.visual.blocks.0.attn.qkv.weight",
            "model.language_model.visual.merger.linear_fc1.weight",
        ] {
            assert!(skip(name), "{name}");
            assert!(is_vision_tower_tensor(name));
        }
        for name in [
            "model.language_model.layers.0.mlp.gate.weight",
            "model.language_model.embed_tokens.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "lm_head.weight",
        ] {
            assert!(!skip(name), "{name}");
        }
    }

    #[test]
    fn text_only_keeps_the_ple_offload_skip() {
        let config = flash_next(true);
        let ple = format!(
            "{}.ple.ple_embedding.ngram_embedding.weight",
            config.layer_prefix(1)
        );
        let both = serving_skip(&config, true).unwrap();
        assert!(both("model.visual.blocks.0.mlp.linear_fc2.weight"));
        let table = format!(
            "{}.ple.ple_embedding.ngram_embedding",
            config.layer_prefix(1)
        );
        assert_eq!(
            both(&ple),
            spark_model::layers::qwen4_ple::skip_offloaded_ple_tensor(&table, &ple)
        );
        let ple_only = serving_skip(&config, false).unwrap();
        assert!(!ple_only("model.visual.blocks.0.mlp.linear_fc2.weight"));
    }

    #[test]
    fn default_and_other_models_load_the_vision_tower() {
        assert!(serving_skip(&flash_next(false), false).is_none());
        let mut other = ModelConfig::qwen3_next_80b_nvfp4();
        other.model_type = "qwen3_5".to_string();
        assert!(serving_skip(&other, true).is_none());
    }

    #[test]
    fn pp_stages_skip_nonlocal_layers_and_non_head_mtp() {
        let mut stage1 = flash_next(false);
        stage1.pp_world_size = 3;
        stage1.pp_rank = 1;
        stage1.pp_layer_offset = 16;
        stage1.num_hidden_layers = 16;
        let skip = serving_skip(&stage1, false).expect("PP>1 installs a layer skip");
        assert!(skip("model.language_model.layers.0.mlp.gate.weight"));
        assert!(skip("model.language_model.layers.15.mlp.gate.weight"));
        assert!(!skip("model.language_model.layers.16.mlp.gate.weight"));
        assert!(!skip("model.language_model.layers.31.mlp.gate.weight"));
        assert!(skip("model.language_model.layers.32.mlp.gate.weight"));
        assert!(skip("mtp.layers.0.self_attn.q_proj.weight"));
        assert!(!skip("model.language_model.embed_tokens.weight"));
        assert!(!skip("lm_head.weight"));

        let mut stage0 = flash_next(false);
        stage0.pp_world_size = 4;
        stage0.pp_rank = 0;
        stage0.pp_layer_offset = 0;
        stage0.num_hidden_layers = 12;
        let skip0 = serving_skip(&stage0, false).expect("PP=4 stage 0 installs a layer skip");
        assert!(!skip0("model.language_model.layers.0.mlp.gate.weight"));
        assert!(!skip0("model.language_model.layers.11.mlp.gate.weight"));
        assert!(skip0("model.language_model.layers.12.mlp.gate.weight"));
        assert!(!skip0("mtp.layers.0.self_attn.q_proj.weight"));
    }
}
