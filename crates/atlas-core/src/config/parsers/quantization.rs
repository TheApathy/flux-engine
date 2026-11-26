// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `config.rs` for file-size budget. Parser for a model family.

#![allow(unused_imports)]

use std::collections::HashSet;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

use super::super::{
    ModelConfig, ModelOptQuantizationGroup, ModelOptWeightFormat, QuantizationConfig,
};

pub fn parse_quantization_config(raw: &serde_json::Value) -> Option<QuantizationConfig> {
    // Kept as an Option-returning compatibility surface for callers that merge
    // optional sidecars after parsing. Architecture admission uses the checked
    // form below and therefore never swallows malformed mixed precision.
    parse_quantization_config_checked(raw).ok().flatten()
}

pub(crate) fn parse_quantization_config_checked(
    raw: &serde_json::Value,
) -> Result<Option<QuantizationConfig>> {
    let Some(qc_raw) = raw.get("quantization_config") else {
        return Ok(None);
    };
    // NVIDIA ModelOpt's sibling `hf_quant_config.json` (merged into the
    // `quantization_config` slot by `merge_sidecar_quant_config`) nests the
    // quant fields one level deep under a `"quantization"` object and never
    // emits a `quant_method` key — the scheme is implied by
    // `producer.name == "modelopt"`:
    //
    //   { "producer": { "name": "modelopt", ... },
    //     "quantization": { "quant_algo": "NVFP4",
    //                       "exclude_modules": ["lm_head", ...] } }
    //
    // Read at the top level this parses to all-empty and returns `None`,
    // which silently drops the checkpoint into tensor-name heuristics — for
    // a ModelOpt NVFP4 checkpoint (no `.weight_packed`, `.mixer.`-prefixed
    // Nemotron modules) those mis-detect as `Bf16Raw` and limp into a
    // runtime BF16->NVFP4 requant path, degrading a fully-calibrated NVFP4
    // release. `normalize_modelopt_sidecar` lifts the nested object and
    // synthesizes `quant_method` so the rest of this parser sees the
    // canonical flat shape. Already-flat configs pass through unchanged.
    let canonical = normalize_modelopt_sidecar(qc_raw);
    let qc = &canonical;

    // Either scheme may set any of these top-level strings:
    //   quant_method     — scheme name; both schemes set this.
    //   quant_algo       — ModelOpt-specific label (e.g. "NVFP4"). Also
    //                      propagated when producer.name=="modelopt".
    //   format           — compressed-tensors only
    //                      (e.g. "nvfp4-pack-quantized").
    let quant_method = qc
        .get("quant_method")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();
    let quant_algo = qc
        .get("quant_algo")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            // ModelOpt dumps sometimes put quant_algo under
            // `config_groups.group_0.weights.type` as `"float"` with a
            // `num_bits` sibling. Mine those for a best-effort label.
            let group = qc.get("config_groups")?.get("group_0")?;
            let weights = group.get("weights")?;
            let bits = weights.get("num_bits")?.as_u64()?;
            let ty = weights.get("type")?.as_str()?;
            match (bits, ty) {
                (4, "float") => Some("NVFP4"),
                (8, "float") => Some("FP8"),
                _ => None,
            }
        })
        .unwrap_or("")
        .to_string();
    let format = qc
        .get("format")
        .or_else(|| qc.get("fmt"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();

    // Ignore list aliases used by ModelOpt, compressed-tensors, and native
    // Transformers FP8 checkpoints respectively. Preserve their order while
    // deduplicating so downstream format dispatch never guesses precision.
    let mut ignore_modules: Vec<String> = Vec::new();
    for key in ["ignore", "exclude_modules", "modules_to_not_convert"] {
        if let Some(arr) = qc.get(key).and_then(serde_json::Value::as_array) {
            for value in arr {
                if let Some(module) = value.as_str()
                    && !ignore_modules.iter().any(|known| known == module)
                {
                    ignore_modules.push(module.to_string());
                }
            }
        }
    }

    // An empty quant_method with empty ignore list is not a real quant
    // config — skip so callers can fall through to heuristic detection.
    if quant_method.is_empty() && quant_algo.is_empty() && ignore_modules.is_empty() {
        return Ok(None);
    }

    let mut config_groups = parse_modelopt_mixed_groups(qc, &quant_method, &quant_algo)?;
    // `parse_modelopt_mixed_groups` only understands the named
    // `config_groups` map (`{"group_0": {"weights": {"type": "float",
    // "num_bits": 4, ...}, "targets": [...]}}`). Some ModelOpt 0.47
    // MIXED_PRECISION sidecars instead list `quantized_layers`, a flatter
    // per-module map (`{"<module path>": {"quant_algo": "NVFP4",
    // "group_size": 16}}` — e.g. the per-expert groups
    // `Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2` ships). Fall back to it only
    // when the named-group parse found nothing, so a sidecar that supplies
    // both shapes keeps using the richer one.
    if config_groups.is_empty() {
        config_groups = parse_modelopt_quantized_layers(qc, &quant_method, &quant_algo)?;
    }

    Ok(Some(QuantizationConfig {
        quant_method,
        quant_algo,
        format,
        ignore_modules,
        config_groups,
    }))
}

fn parse_modelopt_mixed_groups(
    qc: &serde_json::Value,
    quant_method: &str,
    quant_algo: &str,
) -> Result<Vec<ModelOptQuantizationGroup>> {
    if !quant_method.eq_ignore_ascii_case("modelopt")
        || !quant_algo.eq_ignore_ascii_case("MIXED_PRECISION")
    {
        return Ok(Vec::new());
    }

    let Some(groups_value) = qc.get("config_groups") else {
        // Some ModelOpt sidecars advertise only the aggregate algorithm. Keep
        // that metadata parseable; the weight loader still rejects the
        // ambiguous mixed format until the checkpoint supplies typed groups.
        return Ok(Vec::new());
    };
    let groups = groups_value
        .as_object()
        .context("ModelOpt MIXED_PRECISION config_groups must be an object")?;
    if groups.is_empty() {
        bail!("ModelOpt MIXED_PRECISION requires at least one config group");
    }

    let mut names = groups.keys().collect::<Vec<_>>();
    names.sort_unstable();
    let mut seen_targets = HashSet::new();
    let mut parsed = Vec::with_capacity(names.len());

    for name in names {
        let group = groups[name]
            .as_object()
            .with_context(|| format!("ModelOpt config group {name:?} must be an object"))?;
        let weights = group
            .get("weights")
            .and_then(serde_json::Value::as_object)
            .with_context(|| format!("ModelOpt config group {name:?} is missing weights"))?;
        let weight_type = weights
            .get("type")
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("ModelOpt config group {name:?} is missing weights.type"))?;
        if weight_type != "float" {
            bail!("ModelOpt config group {name:?} has unsupported weights.type {weight_type:?}");
        }
        if weights
            .get("dynamic")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            bail!("ModelOpt config group {name:?} uses unsupported dynamic weights");
        }

        let bits = weights
            .get("num_bits")
            .and_then(serde_json::Value::as_u64)
            .with_context(|| {
                format!("ModelOpt config group {name:?} is missing weights.num_bits")
            })?;
        let declared_group_size = modelopt_group_size(weights)?;
        let (weight_format, group_size) = match (bits, declared_group_size) {
            (4, Some(16)) => (ModelOptWeightFormat::Nvfp4, 16),
            (4, Some(group_size)) => bail!(
                "ModelOpt config group {name:?} requires NVFP4 group size 16, got {group_size}"
            ),
            (4, None) => {
                bail!("ModelOpt config group {name:?} is missing its NVFP4 weight group size")
            }
            // ModelOpt's ordinary FP8 E4M3 format is per-tensor scaled and
            // intentionally has no group_size. MXFP8 is a distinct grouped
            // layout with E8M0 scales and an explicit width of 32.
            (8, None) => (ModelOptWeightFormat::Fp8, 0),
            (8, Some(32)) => (ModelOptWeightFormat::Mxfp8, 32),
            (8, Some(group_size)) => bail!(
                "ModelOpt config group {name:?} requires MXFP8 group size 32 when grouped, got {group_size}"
            ),
            _ => bail!("ModelOpt config group {name:?} has unsupported {bits}-bit weights"),
        };

        let target_values = group
            .get("targets")
            .and_then(serde_json::Value::as_array)
            .with_context(|| format!("ModelOpt config group {name:?} is missing targets"))?;
        if target_values.is_empty() {
            bail!("ModelOpt config group {name:?} has no targets");
        }
        let mut targets = Vec::with_capacity(target_values.len());
        for value in target_values {
            let target = value.as_str().with_context(|| {
                format!("ModelOpt config group {name:?} contains a non-string target")
            })?;
            if target.is_empty() {
                bail!("ModelOpt config group {name:?} contains an empty target");
            }
            if !seen_targets.insert(target.to_string()) {
                bail!("ModelOpt mixed precision has duplicate target {target:?}");
            }
            targets.push(target.to_string());
        }

        parsed.push(ModelOptQuantizationGroup {
            name: name.to_string(),
            weight_format,
            group_size,
            targets,
        });
    }

    Ok(parsed)
}

/// Parse ModelOpt's flatter `quantized_layers` per-module map, the
/// `MIXED_PRECISION` shape that has no named `config_groups` (see the call
/// site above). Each entry becomes its own single-target group so
/// `QuantizationConfig::modelopt_weight_format_for` keeps working
/// unchanged. Only called when `quant_algo` is `MIXED_PRECISION` and
/// `config_groups` was absent or empty.
fn parse_modelopt_quantized_layers(
    qc: &serde_json::Value,
    quant_method: &str,
    quant_algo: &str,
) -> Result<Vec<ModelOptQuantizationGroup>> {
    if !quant_method.eq_ignore_ascii_case("modelopt")
        || !quant_algo.eq_ignore_ascii_case("MIXED_PRECISION")
    {
        return Ok(Vec::new());
    }
    let Some(layers_value) = qc.get("quantized_layers") else {
        return Ok(Vec::new());
    };
    let layers = layers_value
        .as_object()
        .context("ModelOpt quantized_layers must be an object")?;
    if layers.is_empty() {
        bail!("ModelOpt MIXED_PRECISION quantized_layers must not be empty");
    }

    let mut names = layers.keys().collect::<Vec<_>>();
    names.sort_unstable();
    let mut parsed = Vec::with_capacity(names.len());
    for name in names {
        let entry = layers[name].as_object().with_context(|| {
            format!("ModelOpt quantized_layers entry {name:?} must be an object")
        })?;
        let algo = entry
            .get("quant_algo")
            .and_then(serde_json::Value::as_str)
            .with_context(|| {
                format!("ModelOpt quantized_layers entry {name:?} is missing quant_algo")
            })?;
        let group_size = entry.get("group_size").and_then(serde_json::Value::as_u64);
        let (weight_format, resolved_group_size) = match (algo, group_size) {
            ("NVFP4" | "W4A16_NVFP4", Some(16)) => (ModelOptWeightFormat::Nvfp4, 16),
            ("NVFP4" | "W4A16_NVFP4", other) => bail!(
                "ModelOpt quantized_layers entry {name:?} requires {algo} group size 16, got {other:?}"
            ),
            ("MXFP8", Some(32)) => (ModelOptWeightFormat::Mxfp8, 32),
            ("MXFP8", other) => bail!(
                "ModelOpt quantized_layers entry {name:?} requires MXFP8 group size 32, got {other:?}"
            ),
            ("FP8", None | Some(0)) => (ModelOptWeightFormat::Fp8, 0),
            ("FP8", Some(other)) => bail!(
                "ModelOpt quantized_layers entry {name:?} requires FP8 group size 0 (or omitted), got {other}"
            ),
            (other, _) => bail!(
                "ModelOpt quantized_layers entry {name:?} has unsupported quant_algo {other:?} \
                 (supported: NVFP4/W4A16_NVFP4 group 16, MXFP8 group 32, FP8 group 0/omitted)"
            ),
        };
        parsed.push(ModelOptQuantizationGroup {
            name: name.to_string(),
            weight_format,
            group_size: resolved_group_size,
            targets: vec![name.to_string()],
        });
    }
    Ok(parsed)
}

fn modelopt_group_size(
    weights: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<usize>> {
    fn positive_usize(value: &serde_json::Value, field: &str) -> Result<usize> {
        let raw = value
            .as_u64()
            .with_context(|| format!("ModelOpt weights.{field} must be a positive integer"))?;
        ensure!(raw > 0, "ModelOpt weights.{field} must be positive");
        usize::try_from(raw)
            .with_context(|| format!("ModelOpt weights.{field} exceeds the platform size limit"))
    }

    let mut declarations: Vec<(&str, usize)> = Vec::new();
    for key in ["group_size", "block_size"] {
        if let Some(value) = weights.get(key) {
            declarations.push((key, positive_usize(value, key)?));
        }
    }

    if let Some(value) = weights.get("block_sizes") {
        let sizes = value
            .as_array()
            .context("ModelOpt weights.block_sizes must be an array of integers")?;
        ensure!(
            !sizes.is_empty(),
            "ModelOpt weights.block_sizes must not be empty"
        );
        for (index, size) in sizes[..sizes.len() - 1].iter().enumerate() {
            // ModelOpt uses -1 as the wildcard for non-grouped outer axes,
            // e.g. `[-1, 16]`. No other negative or malformed dimension is
            // meaningful, and the terminal group width must always be > 0.
            if size.as_i64() == Some(-1) {
                continue;
            }
            positive_usize(size, &format!("block_sizes[{index}]"))?;
        }
        let terminal = positive_usize(
            sizes.last().expect("non-empty block_sizes checked above"),
            &format!("block_sizes[{}]", sizes.len() - 1),
        )?;
        declarations.push(("block_sizes", terminal));
    }

    let Some(&(first_name, first_size)) = declarations.first() else {
        return Ok(None);
    };
    for &(name, size) in &declarations[1..] {
        ensure!(
            size == first_size,
            "conflicting ModelOpt weight group sizes: {first_name}={first_size}, {name}={size}"
        );
    }
    Ok(Some(first_size))
}

/// Flatten a ModelOpt-style `hf_quant_config.json` payload into the canonical
/// shape the rest of [`parse_quantization_config`] reads.
///
/// Two transforms, both no-ops on an already-canonical (HF-standard) block:
///   1. If the payload has a nested `"quantization"` object, lift it to the
///      top level (ModelOpt nests `quant_algo` / `exclude_modules` there).
///   2. If no `quant_method` is present but `producer.name == "modelopt"`,
///      synthesize `quant_method = "modelopt"` so downstream config-first
///      dispatch (`detect_quant_format`, `detect_nvfp4_variant`) recognizes
///      the scheme. An explicit `quant_method` is never overwritten.
fn normalize_modelopt_sidecar(qc_raw: &serde_json::Value) -> serde_json::Value {
    // (1) Lift nested `quantization` object if present.
    let mut canonical = match qc_raw.get("quantization") {
        Some(serde_json::Value::Object(inner)) => serde_json::Value::Object(inner.clone()),
        _ => qc_raw.clone(),
    };

    let Some(obj) = canonical.as_object_mut() else {
        return canonical;
    };

    // (2) Synthesize `quant_method` from `producer.name` when absent.
    let has_method = obj
        .get("quant_method")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| !s.is_empty());
    if !has_method {
        let producer_modelopt = qc_raw
            .get("producer")
            .and_then(|p| p.get("name"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|n| n.eq_ignore_ascii_case("modelopt"));
        if producer_modelopt {
            obj.insert(
                "quant_method".to_string(),
                serde_json::Value::String("modelopt".to_string()),
            );
        }
    }

    canonical
}

#[cfg(test)]
mod tests {
    use super::{parse_quantization_config, parse_quantization_config_checked};
    use crate::config::ModelOptWeightFormat;

    fn mixed_group_with_weights(weights: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "config_groups": {
                    "group_0": {
                        "weights": weights,
                        "targets": ["model.language_model.layers.0.self_attn.q_proj"]
                    }
                }
            }
        })
    }

    /// The exact `hf_quant_config.json` schema shipped by
    /// `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`, wrapped by
    /// `merge_sidecar_quant_config` into the `quantization_config` slot.
    #[test]
    fn modelopt_sidecar_nested_schema_parses() {
        let raw = serde_json::json!({
            "quantization_config": {
                "producer": { "name": "modelopt", "version": "0.29.0" },
                "quantization": {
                    "quant_algo": "NVFP4",
                    "kv_cache_quant_algo": "FP8",
                    "group_size": 16,
                    "exclude_modules": [
                        "lm_head",
                        "backbone.layers.4.mixer.in_proj",
                        "backbone.layers.0.mixer.conv1d"
                    ]
                }
            }
        });
        let qc = parse_quantization_config(&raw)
            .expect("ModelOpt nested sidecar must yield a QuantizationConfig");
        assert_eq!(qc.quant_method, "modelopt");
        assert_eq!(qc.quant_algo, "NVFP4");
        assert_eq!(qc.ignore_modules.len(), 3);
        assert!(qc.ignore_modules.iter().any(|m| m == "lm_head"));
        assert!(
            qc.ignore_modules
                .iter()
                .any(|m| m == "backbone.layers.4.mixer.in_proj")
        );
    }

    /// An already-flat HF-standard block (compressed-tensors) must be
    /// unaffected by the ModelOpt normalization.
    #[test]
    fn flat_compressed_tensors_block_unchanged() {
        let raw = serde_json::json!({
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "ignore": ["lm_head"]
            }
        });
        let qc = parse_quantization_config(&raw).expect("flat block must still parse");
        assert_eq!(qc.quant_method, "compressed-tensors");
        assert_eq!(qc.format, "nvfp4-pack-quantized");
        assert_eq!(qc.ignore_modules, vec!["lm_head".to_string()]);
    }

    /// ModelOpt mixed-precision sidecar (Super-120B) — nested, no
    /// `exclude_modules`. Must still resolve `quant_method = modelopt`.
    #[test]
    fn modelopt_mixed_precision_sidecar_parses() {
        let raw = serde_json::json!({
            "quantization_config": {
                "producer": { "name": "modelopt", "version": "0.43.0" },
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "kv_cache_quant_algo": "FP8"
                }
            }
        });
        let qc = parse_quantization_config(&raw)
            .expect("mixed-precision sidecar must yield a QuantizationConfig");
        assert_eq!(qc.quant_method, "modelopt");
        assert_eq!(qc.quant_algo, "MIXED_PRECISION");
    }

    #[test]
    fn modelopt_mixed_precision_groups_are_typed() {
        let raw = serde_json::json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "config_groups": {
                    "nvfp4_experts": {
                        "weights": {
                            "type": "float",
                            "num_bits": 4,
                            "block_sizes": [-1, 16],
                            "dynamic": false
                        },
                        "targets": ["re:.*mlp.experts.*"]
                    },
                    "mxfp8_attention": {
                        "weights": {
                            "type": "float",
                            "num_bits": 8,
                            "block_sizes": [-1, 32],
                            "dynamic": false
                        },
                        "targets": ["re:.*self_attn.*", "re:.*visual.*"]
                    }
                }
            }
        });

        let qc = parse_quantization_config_checked(&raw)
            .expect("valid mixed ModelOpt config must parse")
            .expect("quantization config must be present");
        assert_eq!(qc.config_groups.len(), 2);
        assert_eq!(qc.config_groups[0].name, "mxfp8_attention");
        assert_eq!(
            qc.config_groups[0].weight_format,
            ModelOptWeightFormat::Mxfp8
        );
        assert_eq!(qc.config_groups[0].group_size, 32);
        assert_eq!(
            qc.config_groups[1].weight_format,
            ModelOptWeightFormat::Nvfp4
        );
        assert_eq!(qc.config_groups[1].group_size, 16);
        assert_eq!(
            qc.modelopt_weight_format_for("re:.*self_attn.*"),
            Some(ModelOptWeightFormat::Mxfp8)
        );
        assert_eq!(qc.modelopt_weight_format_for("not-listed"), None);
    }

    #[test]
    fn modelopt_mixed_standard_fp8_without_group_size_is_typed() {
        let raw = serde_json::json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "type": "float",
                            "num_bits": 8,
                            "dynamic": false
                        },
                        "targets": ["model.language_model.layers.0.self_attn.q_proj"]
                    },
                    "group_1": {
                        "weights": {
                            "type": "float",
                            "num_bits": 4,
                            "group_size": 16,
                            "dynamic": false
                        },
                        "targets": ["model.language_model.layers.0.mlp.gate_proj"]
                    }
                }
            }
        });

        let qc = parse_quantization_config_checked(&raw)
            .expect("ModelOpt FP8 plus NVFP4 config must parse")
            .expect("quantization config must be present");
        assert_eq!(qc.config_groups[0].weight_format, ModelOptWeightFormat::Fp8);
        assert_eq!(qc.config_groups[0].group_size, 0);
        assert_eq!(
            qc.config_groups[1].weight_format,
            ModelOptWeightFormat::Nvfp4
        );
        assert_eq!(qc.config_groups[1].group_size, 16);
    }

    #[test]
    fn modelopt_mixed_nvfp4_without_group_size_fails_closed() {
        let raw = serde_json::json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "config_groups": {
                    "group_0": {
                        "weights": { "type": "float", "num_bits": 4, "dynamic": false },
                        "targets": ["model.language_model.layers.0.mlp.gate_proj"]
                    }
                }
            }
        });

        let err = parse_quantization_config_checked(&raw)
            .expect_err("NVFP4 without its width must fail closed");
        assert!(
            err.to_string()
                .contains("missing its NVFP4 weight group size")
        );
    }

    #[test]
    fn modelopt_mixed_precision_rejects_wrong_group_size() {
        let raw = serde_json::json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "config_groups": {
                    "bad_mxfp8": {
                        "weights": {
                            "type": "float",
                            "num_bits": 8,
                            "block_sizes": [-1, 16],
                            "dynamic": false
                        },
                        "targets": ["re:.*self_attn.*"]
                    }
                }
            }
        });

        let err = parse_quantization_config_checked(&raw)
            .expect_err("MXFP8 with group size 16 must fail closed");
        assert!(err.to_string().contains("MXFP8 group size 32"));
    }

    #[test]
    fn modelopt_group_size_rejects_malformed_negative_and_empty_values() {
        for weights in [
            serde_json::json!({"type":"float", "num_bits":8, "group_size":"32"}),
            serde_json::json!({"type":"float", "num_bits":8, "group_size":-32}),
            serde_json::json!({"type":"float", "num_bits":8, "group_size":0}),
            serde_json::json!({"type":"float", "num_bits":8, "block_sizes":[]}),
            serde_json::json!({"type":"float", "num_bits":8, "block_sizes":[-2,32]}),
            serde_json::json!({"type":"float", "num_bits":8, "block_sizes":[-1,-32]}),
            serde_json::json!({"type":"float", "num_bits":8, "block_sizes":[-1,"32"]}),
        ] {
            let err = parse_quantization_config_checked(&mixed_group_with_weights(weights))
                .expect_err("malformed ModelOpt group width must fail closed");
            assert!(
                err.to_string().contains("positive")
                    || err.to_string().contains("must not be empty"),
                "unexpected error: {err:#}"
            );
        }
    }

    #[test]
    fn modelopt_group_size_rejects_overflow_and_conflicting_spellings() {
        let overflow: serde_json::Value =
            serde_json::from_str(r#"{"type":"float","num_bits":8,"group_size":1e100}"#)
                .expect("large JSON number must remain representable as a Value");
        assert!(
            parse_quantization_config_checked(&mixed_group_with_weights(overflow)).is_err(),
            "out-of-range group width must fail closed"
        );

        for weights in [
            serde_json::json!({
                "type":"float", "num_bits":4,
                "group_size":16, "block_size":32
            }),
            serde_json::json!({
                "type":"float", "num_bits":4,
                "group_size":16, "block_sizes":[-1,32]
            }),
        ] {
            let err = parse_quantization_config_checked(&mixed_group_with_weights(weights))
                .expect_err("conflicting ModelOpt group widths must fail closed");
            assert!(
                err.to_string()
                    .contains("conflicting ModelOpt weight group sizes")
            );
        }
    }

    #[test]
    fn modelopt_mixed_precision_rejects_duplicate_targets() {
        let raw = serde_json::json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "config_groups": {
                    "experts_a": {
                        "weights": {
                            "type": "float",
                            "num_bits": 4,
                            "group_size": 16,
                            "dynamic": false
                        },
                        "targets": ["re:.*mlp.experts.*"]
                    },
                    "experts_b": {
                        "weights": {
                            "type": "float",
                            "num_bits": 4,
                            "group_size": 16,
                            "dynamic": false
                        },
                        "targets": ["re:.*mlp.experts.*"]
                    }
                }
            }
        });

        let err = parse_quantization_config_checked(&raw)
            .expect_err("duplicate mixed-precision targets must fail closed");
        assert!(err.to_string().contains("duplicate target"));
    }

    #[test]
    fn transformers_fp8_modules_to_not_convert_are_preserved() {
        let raw = serde_json::json!({
            "quantization_config": {
                "quant_method": "fp8",
                "modules_to_not_convert": [
                    "model.layers.0.self_attn.q_proj",
                    "model.layers.0.self_attn.q_proj",
                    "lm_head"
                ]
            }
        });
        let qc = parse_quantization_config(&raw).expect("native FP8 block must parse");
        assert_eq!(qc.quant_method, "fp8");
        assert_eq!(
            qc.ignore_modules,
            vec![
                "model.layers.0.self_attn.q_proj".to_string(),
                "lm_head".to_string()
            ]
        );
    }

    /// The `hf_quant_config.json` shape ModelOpt 0.47.0rc0 writes for
    /// `Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2`, abbreviated to 3 of its
    /// 48 `quantized_layers` entries and 2 of its `exclude_modules`: no
    /// `config_groups` key, `quant_algo = "MIXED_PRECISION"`, and a flat
    /// per-module `quantized_layers` map instead. Every real entry in that
    /// file is NVFP4/16 — this must resolve to `config_groups` entries
    /// `modelopt_weight_format_for` can answer, which is what
    /// `spark_server::serve_phases::weights::quant_multiplier` needs.
    #[test]
    fn modelopt_mixed_precision_quantized_layers_shape_parses() {
        let raw = serde_json::json!({
            "quantization_config": {
                "producer": { "name": "modelopt", "version": "0.47.0rc0" },
                "quantization": {
                    "group_size": 16,
                    "kv_cache_quant_algo": null,
                    "quant_algo": "MIXED_PRECISION",
                    "exclude_modules": [
                        "lm_head",
                        "model.language_model.layers.0.self_attn*"
                    ],
                    "quantized_layers": {
                        "model.language_model.layers.0.mlp.experts": {
                            "group_size": 16, "quant_algo": "NVFP4"
                        },
                        "model.language_model.layers.1.mlp.experts": {
                            "group_size": 16, "quant_algo": "NVFP4"
                        },
                        "model.language_model.layers.47.mlp.experts": {
                            "group_size": 16, "quant_algo": "NVFP4"
                        }
                    }
                }
            }
        });
        let qc = parse_quantization_config(&raw)
            .expect("CyberFrost's quantized_layers shape must parse");
        assert_eq!(qc.quant_method, "modelopt");
        assert_eq!(qc.quant_algo, "MIXED_PRECISION");
        assert_eq!(qc.ignore_modules.len(), 2);
        assert_eq!(qc.config_groups.len(), 3);
        assert!(
            qc.config_groups
                .iter()
                .all(|g| g.weight_format == ModelOptWeightFormat::Nvfp4 && g.group_size == 16)
        );
        assert_eq!(
            qc.modelopt_weight_format_for("model.language_model.layers.1.mlp.experts"),
            Some(ModelOptWeightFormat::Nvfp4)
        );
        assert_eq!(
            qc.modelopt_weight_format_for("model.language_model.layers.2.mlp.experts"),
            None,
            "a module absent from quantized_layers must not resolve"
        );
    }

    /// `config_groups` takes precedence when a sidecar supplies both
    /// shapes — `parse_modelopt_quantized_layers` is a fallback for when
    /// the named-group parse finds nothing, not an override.
    #[test]
    fn named_config_groups_take_precedence_over_quantized_layers() {
        let raw = serde_json::json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "config_groups": {
                    "group_0": {
                        "weights": { "type": "float", "num_bits": 4, "group_size": 16 },
                        "targets": ["model.layers.0.mlp.experts"]
                    }
                },
                "quantized_layers": {
                    "model.layers.1.mlp.experts": { "group_size": 16, "quant_algo": "NVFP4" }
                }
            }
        });
        let qc = parse_quantization_config(&raw).expect("dual-shape sidecar must parse");
        assert_eq!(qc.config_groups.len(), 1);
        assert_eq!(
            qc.config_groups[0].targets,
            vec!["model.layers.0.mlp.experts".to_string()]
        );
    }

    /// A `quantized_layers` entry naming an unsupported algorithm must
    /// fail closed rather than silently resolving as NVFP4, while `FP8`
    /// with omitted `group_size` is accepted as ungrouped per-tensor FP8.
    #[test]
    fn quantized_layers_rejects_unsupported_quant_algo() {
        let fp8_omitted = serde_json::json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "model.layers.0.mlp.experts": { "group_size": 16, "quant_algo": "NVFP4" },
                    "model.layers.1.self_attn": { "quant_algo": "FP8" }
                }
            }
        });
        let qc = parse_quantization_config_checked(&fp8_omitted)
            .expect("FP8 with omitted group_size must parse as ungrouped FP8")
            .expect("quantization_config must be Some");
        assert_eq!(
            qc.modelopt_weight_format_for("model.layers.1.self_attn"),
            Some(ModelOptWeightFormat::Fp8)
        );

        let unsupported = serde_json::json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "model.layers.0.mlp.experts": { "group_size": 16, "quant_algo": "NVFP4" },
                    "model.layers.1.self_attn": { "quant_algo": "INT4_AWQ" }
                }
            }
        });
        let err = parse_quantization_config_checked(&unsupported)
            .expect_err("an unsupported quantized_layers quant_algo must fail closed");
        assert!(err.to_string().contains("unsupported quant_algo"));
    }

    #[test]
    fn quantized_layers_admits_qad5500_hybrid_nvfp4_w4a16_and_mxfp8() {
        let raw = serde_json::json!({
            "quantization_config": {
                "producer": { "name": "modelopt", "version": "0.42.0" },
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {
                        "model.language_model.layers.0.mlp.experts": {
                            "group_size": 16,
                            "quant_algo": "NVFP4"
                        },
                        "model.language_model.layers.0.linear_attn.in_proj_qkv": {
                            "group_size": 32,
                            "quant_algo": "MXFP8"
                        },
                        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding": {
                            "group_size": 16,
                            "quant_algo": "NVFP4"
                        },
                        "mtp.layers.0.mlp.experts": {
                            "group_size": 16,
                            "quant_algo": "W4A16_NVFP4"
                        }
                    }
                }
            }
        });
        let qc = parse_quantization_config_checked(&raw)
            .expect("QAD5500-Hybrid quantized_layers must parse")
            .expect("quantization config must be present");
        assert_eq!(
            qc.modelopt_weight_format_for("model.layers.0.mlp.experts.511.down_proj"),
            Some(ModelOptWeightFormat::Nvfp4)
        );
        assert_eq!(
            qc.modelopt_weight_format_for("model.layers.0.linear_attn.in_proj_qkv"),
            Some(ModelOptWeightFormat::Mxfp8)
        );
        assert_eq!(
            qc.modelopt_weight_format_for("model.layers.1.ple.ple_embedding.ngram_embedding"),
            Some(ModelOptWeightFormat::Nvfp4)
        );
        assert_eq!(
            qc.modelopt_weight_format_for("mtp.layers.0.mlp.experts.0.gate_proj"),
            Some(ModelOptWeightFormat::Nvfp4)
        );
    }
}
