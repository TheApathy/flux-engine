// SPDX-License-Identifier: AGPL-3.0-only

//! Exact BF16 tensor census for an exact `[[dflash.profile]]` DFlash2 row.

use std::collections::BTreeSet;

use anyhow::{Result, ensure};
use atlas_kernels::DrafterProfile;

use super::config::Dimensions;
use super::{TensorMetadataSource, require_tensor};
use crate::weight_loader::dflash_loader::DflashConfig;

pub(super) fn has_tensor_signature(source: &dyn TensorMetadataSource) -> bool {
    source.names().any(|name| {
        let name = name.strip_prefix("model.").unwrap_or(name);
        name.starts_with("candidate_selector.")
            || (name.starts_with("layers.")
                && (name.contains(".attention_conv.") || name.contains(".mlp_conv.")))
    })
}

pub(super) fn validate_official_tensors(
    source: &dyn TensorMetadataSource,
    prefix: &str,
    config: &DflashConfig,
    dim: Dimensions,
    profile: &DrafterProfile,
) -> Result<()> {
    let label = profile.label();
    let id = &profile.drafter;
    let (Some(conv_kernel), Some(conv_group), Some(selector_rank), Some(tensor_count)) = (
        id.conv_kernel_size,
        id.conv_group_size,
        id.selector_rank,
        id.tensor_count,
    ) else {
        anyhow::bail!(
            "drafter profile {label} must pin conv_kernel_size, conv_group_size, selector_rank \
             and tensor_count for its exact DFlash2 tensor census"
        );
    };
    ensure!(
        config.quantization_config.is_none(),
        "BF16 DFlash2 checkpoint for profile {label} must not declare quantization_config"
    );
    let mut expected = BTreeSet::new();
    add(&mut expected, format!("{prefix}fc.weight"))?;
    add(&mut expected, format!("{prefix}hidden_norm.weight"))?;
    add(&mut expected, format!("{prefix}norm.weight"))?;

    let groups = dim
        .hidden
        .checked_div(conv_group)
        .ok_or_else(|| anyhow::anyhow!("official DFlash2 conv group division failed"))?;
    let projection_rows = 2usize
        .checked_mul(conv_kernel)
        .and_then(|value| value.checked_mul(groups))
        .ok_or_else(|| anyhow::anyhow!("official DFlash2 conv projection width overflow"))?;

    for layer in 0..config.num_hidden_layers {
        let lp = format!("{prefix}layers.{layer}");
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            add(&mut expected, format!("{lp}.{suffix}"))?;
        }
        for stem in ["attention_conv", "mlp_conv"] {
            let base = format!("{lp}.{stem}.base_kernel");
            let projection = format!("{lp}.{stem}.kernel_projection.weight");
            add(&mut expected, base.clone())?;
            add(&mut expected, projection.clone())?;
            require_tensor(source, &base, &[2, conv_kernel, dim.hidden])?;
            require_tensor(source, &projection, &[projection_rows, dim.hidden])?;
        }
    }

    let hidden_projection = format!("{prefix}candidate_selector.hidden_projection.weight");
    let predecessor = format!("{prefix}candidate_selector.predecessor_codebook");
    let successor = format!("{prefix}candidate_selector.successor_codebook");
    for name in [&hidden_projection, &predecessor, &successor] {
        add(&mut expected, name.as_str().to_owned())?;
    }
    require_tensor(source, &hidden_projection, &[selector_rank, dim.hidden])?;
    require_tensor(source, &predecessor, &[config.vocab_size, selector_rank])?;
    require_tensor(source, &successor, &[config.vocab_size, selector_rank])?;

    let mut expected_count = tensor_count;
    if profile.allow_markov && config.markov_rank > 0 {
        let w1 = format!("{prefix}markov_head.markov_w1.weight");
        let w2 = format!("{prefix}markov_head.markov_w2.weight");
        add(&mut expected, w1.clone())?;
        add(&mut expected, w2.clone())?;
        require_tensor(source, &w1, &[config.vocab_size, config.markov_rank])?;
        require_tensor(source, &w2, &[config.vocab_size, config.markov_rank])?;
        expected_count = expected_count
            .checked_add(2)
            .ok_or_else(|| anyhow::anyhow!("official DFlash2 tensor-count overflow"))?;
    }

    ensure!(
        expected.len() == expected_count,
        "drafter profile {label} tensor_count={expected_count} disagrees with its own \
         {}-tensor census",
        expected.len()
    );
    let observed = source.names().map(str::to_owned).collect::<BTreeSet<_>>();
    ensure!(
        observed == expected,
        "drafter profile {label} requires the exact {expected_count}-tensor census; \
         missing={:?}, extra={:?}",
        expected.difference(&observed).collect::<Vec<_>>(),
        observed.difference(&expected).collect::<Vec<_>>()
    );
    Ok(())
}

pub(super) fn add(expected: &mut BTreeSet<String>, name: String) -> Result<()> {
    ensure!(
        expected.insert(name.clone()),
        "duplicate official DFlash2 tensor name `{name}`"
    );
    Ok(())
}
