// SPDX-License-Identifier: AGPL-3.0-only

//! Exact structural identity for `quorum` drafter profiles.
//!
//! The quorum rows are the native Qwen3.8-Flash-Next V3 and DFlash2 drafters
//! (`kernels/gb10/qwen3.8-flash-next/MODEL.toml`). A claimed checkpoint must
//! match its row's config identity, its exact bare-name BF16 tensor manifest
//! and its structural parameter count; the same manifest is the pre-upload
//! finiteness receipt.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, ensure};
use atlas_kernels::{DrafterKind, DrafterProfile};

use super::profile;
use super::{TensorMetadataSource, require_tensor};
use crate::weight_loader::dflash_loader::DflashConfig;

/// Quorum rows that claim this checkpoint by its config alone.
pub(super) fn quorum_claimants(config: &DflashConfig) -> Vec<&'static DrafterProfile> {
    profile::quorum_claimants(config)
}

fn validate_native_config(
    config: &DflashConfig,
    claimants: &[&'static DrafterProfile],
) -> Result<&'static DrafterProfile> {
    // Every DFlash/DFlash2 quorum row is a bidirectional block-diffusion
    // drafter; an omitted `is_causal` must never fall back to the layer-type
    // heuristic. All-full-attention DSpark quorum rows omit `is_causal`.
    let requires_explicit_noncausal = claimants
        .iter()
        .all(|profile| profile.drafter.is_causal == Some(false))
        || claimants.iter().any(|profile| {
            profile.drafter.is_causal == Some(false)
                && profile.drafter.architecture == config.architectures.first().map(String::as_str)
        });
    if requires_explicit_noncausal {
        ensure!(
            config.is_causal == Some(false),
            "native Flash-Next drafter is_causal must be explicitly false"
        );
    }
    profile::resolve_claimed(config, claimants)
}

pub(super) fn validate_config_if_candidate(config: &DflashConfig) -> Result<()> {
    let claimants = quorum_claimants(config);
    if !claimants.is_empty() {
        validate_native_config(config, &claimants)?;
    }
    Ok(())
}

fn expected_manifest(
    profile: &DrafterProfile,
    config: &DflashConfig,
    prefix: &str,
) -> Result<BTreeMap<String, Vec<usize>>> {
    let id = &profile.drafter;
    let label = profile.label();
    let pinned = |name: &str, value: Option<usize>| {
        value.ok_or_else(|| anyhow::anyhow!("quorum profile {label} must pin {name}"))
    };
    let hidden = pinned("hidden_size", id.hidden_size)?;
    let intermediate = pinned("intermediate_size", id.intermediate_size)?;
    let layers = pinned("num_hidden_layers", id.num_hidden_layers)?;
    let query = pinned("num_attention_heads", id.num_attention_heads)?;
    let kv = pinned("num_key_value_heads", id.num_key_value_heads)?;
    let head = pinned("head_dim", id.head_dim)?;
    let vocab = pinned("vocab_size", id.vocab_size)?;
    let taps = id
        .target_layer_ids
        .ok_or_else(|| anyhow::anyhow!("quorum profile {label} must pin target_layer_ids"))?
        .len();

    let mut expected = BTreeMap::from([
        (format!("{prefix}fc.weight"), vec![hidden, hidden * taps]),
        (format!("{prefix}hidden_norm.weight"), vec![hidden]),
        (format!("{prefix}norm.weight"), vec![hidden]),
    ]);
    let per_layer = [
        ("input_layernorm.weight", vec![hidden]),
        ("post_attention_layernorm.weight", vec![hidden]),
        ("self_attn.q_proj.weight", vec![query * head, hidden]),
        ("self_attn.k_proj.weight", vec![kv * head, hidden]),
        ("self_attn.v_proj.weight", vec![kv * head, hidden]),
        ("self_attn.o_proj.weight", vec![hidden, query * head]),
        ("self_attn.q_norm.weight", vec![head]),
        ("self_attn.k_norm.weight", vec![head]),
        ("mlp.gate_proj.weight", vec![intermediate, hidden]),
        ("mlp.up_proj.weight", vec![intermediate, hidden]),
        ("mlp.down_proj.weight", vec![hidden, intermediate]),
    ];
    let dflash2 = profile.kind == DrafterKind::Dflash2;
    let (conv_kernel, conv_group, selector_rank) = if dflash2 {
        (
            pinned("conv_kernel_size", id.conv_kernel_size)?,
            pinned("conv_group_size", id.conv_group_size)?,
            pinned("selector_rank", id.selector_rank)?,
        )
    } else {
        (0, 1, 0)
    };
    for layer in 0..layers {
        for (suffix, shape) in &per_layer {
            expected.insert(format!("{prefix}layers.{layer}.{suffix}"), shape.clone());
        }
        if dflash2 {
            let groups = hidden / conv_group;
            for sublayer in ["attention_conv", "mlp_conv"] {
                expected.insert(
                    format!("{prefix}layers.{layer}.{sublayer}.base_kernel"),
                    vec![2, conv_kernel, hidden],
                );
                expected.insert(
                    format!("{prefix}layers.{layer}.{sublayer}.kernel_projection.weight"),
                    vec![2 * conv_kernel * groups, hidden],
                );
            }
        }
    }
    if dflash2 {
        expected.insert(
            format!("{prefix}candidate_selector.hidden_projection.weight"),
            vec![selector_rank, hidden],
        );
        expected.insert(
            format!("{prefix}candidate_selector.predecessor_codebook"),
            vec![vocab, selector_rank],
        );
        expected.insert(
            format!("{prefix}candidate_selector.successor_codebook"),
            vec![vocab, selector_rank],
        );
    }
    if profile.allow_markov && config.markov_rank > 0 {
        expected.insert(
            format!("{prefix}markov_head.markov_w1.weight"),
            vec![vocab, config.markov_rank],
        );
        expected.insert(
            format!("{prefix}markov_head.markov_w2.weight"),
            vec![vocab, config.markov_rank],
        );
    }
    Ok(expected)
}

pub(super) fn finiteness_manifest_if_candidate(
    config: &DflashConfig,
) -> Result<Option<BTreeMap<String, Vec<usize>>>> {
    let claimants = quorum_claimants(config);
    if claimants.is_empty() {
        return Ok(None);
    }
    let profile = validate_native_config(config, &claimants)?;
    expected_manifest(profile, config, "").map(Some)
}

pub(super) fn validate_store_if_candidate(
    source: &dyn TensorMetadataSource,
    prefix: &str,
    config: &DflashConfig,
) -> Result<()> {
    // A store whose `fc.weight` has a quorum row's [H, H * taps] shape is
    // claimed even if its config claims nothing.
    let mut claimants = source
        .metadata(&format!("{prefix}fc.weight"))
        .map(|tensor| profile::quorum_fc_claimants(tensor.shape))
        .unwrap_or_default();
    for claimant in quorum_claimants(config) {
        if !claimants.iter().any(|known| std::ptr::eq(*known, claimant)) {
            claimants.push(claimant);
        }
    }
    if claimants.is_empty() {
        return Ok(());
    }
    let profile = validate_native_config(config, &claimants)?;
    let label = profile.label();
    let id = &profile.drafter;
    ensure!(
        !id.bare_tensor_names || prefix.is_empty(),
        "drafter profile {label} tensors must use the canonical bare names"
    );
    let expected = expected_manifest(profile, config, prefix)?;
    let observed: BTreeSet<_> = source.names().map(str::to_owned).collect();
    let expected_names: BTreeSet<_> = expected.keys().cloned().collect();
    let missing: Vec<_> = expected_names.difference(&observed).cloned().collect();
    let unexpected: Vec<_> = observed.difference(&expected_names).cloned().collect();
    ensure!(
        missing.is_empty() && unexpected.is_empty(),
        "drafter profile {label} tensor manifest mismatch: missing={missing:?}, unexpected={unexpected:?}"
    );
    let markov_tensors = if profile.allow_markov && config.markov_rank > 0 {
        2
    } else {
        0
    };
    if let Some(count) = id.tensor_count {
        let expected_count = count
            .checked_add(markov_tensors)
            .ok_or_else(|| anyhow::anyhow!("native tensor-count overflow"))?;
        ensure!(
            expected.len() == expected_count,
            "drafter profile {label} tensor_count={expected_count} disagrees with its own {}-tensor manifest",
            expected.len()
        );
    }
    let mut parameters = 0usize;
    for (name, shape) in &expected {
        require_tensor(source, name, shape)?;
        let elements = shape.iter().try_fold(1usize, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or_else(|| anyhow::anyhow!("native tensor `{name}` size overflow"))
        })?;
        parameters = parameters
            .checked_add(elements)
            .ok_or_else(|| anyhow::anyhow!("native parameter-count overflow"))?;
    }
    if let Some(base_parameters) = id.parameter_count {
        let markov_params = if profile.allow_markov && config.markov_rank > 0 {
            config
                .vocab_size
                .checked_mul(config.markov_rank)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| anyhow::anyhow!("native Markov parameter-count overflow"))?
        } else {
            0
        };
        let expected_parameters = base_parameters
            .checked_add(markov_params)
            .ok_or_else(|| anyhow::anyhow!("native parameter-count overflow"))?;
        ensure!(
            parameters == expected_parameters,
            "drafter profile {label} has {parameters} structural parameters; expected {expected_parameters}"
        );
    }
    Ok(())
}
