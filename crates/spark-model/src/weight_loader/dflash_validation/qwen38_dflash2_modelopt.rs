// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeSet;

use anyhow::{Result, ensure};
use spark_runtime::weights::WeightDtype;

use super::config::Dimensions;
use super::qwen38_dflash2::add;
use super::{TensorMetadataSource, require_tensor};
use crate::weight_loader::dflash_loader::DflashConfig;

const MODELOPT_TENSOR_COUNT: usize = 186;

pub(super) fn has_tensor_signature(source: &dyn TensorMetadataSource) -> bool {
    source.names().any(|name| {
        let name = name.strip_prefix("model.").unwrap_or(name);
        name.starts_with("layers.")
            && (name.ends_with(".weight_scale")
                || name.ends_with(".weight_scale_2")
                || name.ends_with(".input_scale"))
    })
}

pub(super) fn validate_tensors(
    source: &dyn TensorMetadataSource,
    prefix: &str,
    config: &DflashConfig,
    dim: Dimensions,
) -> Result<()> {
    let quant = config
        .quantization_config
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("ModelOpt DFlash2 tensors require quantization_config"))?;
    ensure!(
        quant.quant_method == "modelopt"
            && quant.quant_algo.eq_ignore_ascii_case("NVFP4")
            && quant.group_size == 16,
        "DFlash2 packed projections require ModelOpt NVFP4 group_size=16"
    );

    let mut expected = BTreeSet::new();
    for (name, shape) in [
        (
            format!("{prefix}fc.weight"),
            vec![dim.hidden, dim.target_stack],
        ),
        (format!("{prefix}hidden_norm.weight"), vec![dim.hidden]),
        (format!("{prefix}norm.weight"), vec![dim.hidden]),
    ] {
        add(&mut expected, name.clone())?;
        require_tensor(source, &name, &shape)?;
    }

    let groups = dim
        .hidden
        .checked_div(16)
        .ok_or_else(|| anyhow::anyhow!("official DFlash2 conv group division failed"))?;
    let projection_rows = 4usize
        .checked_mul(groups)
        .ok_or_else(|| anyhow::anyhow!("official DFlash2 conv projection width overflow"))?;
    for layer in 0..config.num_hidden_layers {
        add_layer_tensors(source, &mut expected, prefix, layer, dim, projection_rows)?;
    }

    for (name, shape) in [
        (
            format!("{prefix}candidate_selector.hidden_projection.weight"),
            vec![256, dim.hidden],
        ),
        (
            format!("{prefix}candidate_selector.predecessor_codebook"),
            vec![config.vocab_size, 256],
        ),
        (
            format!("{prefix}candidate_selector.successor_codebook"),
            vec![config.vocab_size, 256],
        ),
    ] {
        add(&mut expected, name.clone())?;
        require_tensor(source, &name, &shape)?;
    }

    ensure!(
        expected.len() == MODELOPT_TENSOR_COUNT,
        "internal ModelOpt DFlash2 tensor census is not {MODELOPT_TENSOR_COUNT}"
    );
    let observed = source.names().map(str::to_owned).collect::<BTreeSet<_>>();
    ensure!(
        observed == expected,
        "Qwen3.8 ModelOpt DFlash2 requires the exact {MODELOPT_TENSOR_COUNT}-tensor census; missing={:?}, extra={:?}",
        expected.difference(&observed).collect::<Vec<_>>(),
        observed.difference(&expected).collect::<Vec<_>>()
    );
    Ok(())
}

fn add_layer_tensors(
    source: &dyn TensorMetadataSource,
    expected: &mut BTreeSet<String>,
    prefix: &str,
    layer: usize,
    dim: Dimensions,
    projection_rows: usize,
) -> Result<()> {
    let lp = format!("{prefix}layers.{layer}");
    for (suffix, shape) in [
        ("input_layernorm.weight", vec![dim.hidden]),
        ("post_attention_layernorm.weight", vec![dim.hidden]),
        ("self_attn.q_norm.weight", vec![dim.head]),
        ("self_attn.k_norm.weight", vec![dim.head]),
    ] {
        let name = format!("{lp}.{suffix}");
        add(expected, name.clone())?;
        require_tensor(source, &name, &shape)?;
    }
    for (stem, n, k) in [
        ("self_attn.q_proj", dim.query, dim.hidden),
        ("self_attn.k_proj", dim.key_value, dim.hidden),
        ("self_attn.v_proj", dim.key_value, dim.hidden),
        ("self_attn.o_proj", dim.hidden, dim.query),
        ("mlp.gate_proj", dim.intermediate, dim.hidden),
        ("mlp.up_proj", dim.intermediate, dim.hidden),
        ("mlp.down_proj", dim.hidden, dim.intermediate),
    ] {
        validate_projection(source, expected, &format!("{lp}.{stem}"), n, k)?;
    }
    for stem in ["attention_conv", "mlp_conv"] {
        for (suffix, shape) in [
            ("base_kernel", vec![2, 2, dim.hidden]),
            (
                "kernel_projection.weight",
                vec![projection_rows, dim.hidden],
            ),
        ] {
            let name = format!("{lp}.{stem}.{suffix}");
            add(expected, name.clone())?;
            require_tensor(source, &name, &shape)?;
        }
    }
    Ok(())
}

fn validate_projection(
    source: &dyn TensorMetadataSource,
    expected: &mut BTreeSet<String>,
    stem: &str,
    n: usize,
    k: usize,
) -> Result<()> {
    ensure!(
        k.is_multiple_of(16),
        "ModelOpt input width is not divisible by 16"
    );
    for (suffix, shape, dtype) in [
        ("weight", vec![n, k / 2], WeightDtype::UInt8),
        ("weight_scale", vec![n, k / 16], WeightDtype::FP8E4M3),
        ("weight_scale_2", vec![], WeightDtype::FP32),
        ("input_scale", vec![], WeightDtype::FP32),
    ] {
        let name = format!("{stem}.{suffix}");
        add(expected, name.clone())?;
        let tensor = source
            .metadata(&name)
            .ok_or_else(|| anyhow::anyhow!("DFlash checkpoint missing required tensor `{name}`"))?;
        ensure!(
            tensor.dtype == dtype && tensor.shape == shape,
            "DFlash ModelOpt tensor `{name}` has {:?} {:?}; expected {:?} {:?}",
            tensor.dtype,
            tensor.shape,
            dtype,
            shape
        );
    }
    Ok(())
}
