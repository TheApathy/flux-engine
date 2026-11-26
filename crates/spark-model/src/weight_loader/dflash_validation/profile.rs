// SPDX-License-Identifier: AGPL-3.0-only

//! Drafter admission against the MODEL.toml `[[dflash.profile]]` table.
//!
//! The table replaced two hand-written validators (the strict Qwen3.8-27B
//! DFlash2 identity and the native Flash-Next constants). `validate_identity`
//! is the one field-by-field matcher both now run through; which rows claim
//! a checkpoint is decided by the row's `admission`.

use anyhow::{Result, bail, ensure};
use atlas_kernels::{DrafterAdmission, DrafterKind, DrafterProfile, VocabPin};

use crate::weight_loader::dflash_loader::{DflashConfig, DrafterCheckpointFamily};

pub(crate) fn profiles() -> &'static [DrafterProfile] {
    atlas_kernels::drafter_profiles()
}

fn rows(admission: DrafterAdmission) -> impl Iterator<Item = &'static DrafterProfile> {
    profiles()
        .iter()
        .filter(move |profile| profile.admission == admission)
}

/// Whether a `quorum` row claims this checkpoint. Two of three identity axes
/// catch any one-field mutation without claiming every future drafter that
/// shares one axis; a DFlash2 row that pins a selector vocabulary also claims
/// any checkpoint of its architecture that declares one.
fn quorum_claims(config: &DflashConfig, profile: &DrafterProfile) -> bool {
    let id = &profile.drafter;
    let target_depth = id.num_target_layers == Some(config.num_target_layers);
    let draft_geometry = id.hidden_size == Some(config.hidden_size)
        && id.intermediate_size == Some(config.intermediate_size)
        && id.num_hidden_layers == Some(config.num_hidden_layers);
    let capture_taps = config
        .dflash_config
        .as_ref()
        .is_some_and(|sub| id.target_layer_ids == Some(sub.target_layer_ids.as_slice()));
    let selector_marker = matches!(id.selector_vocab_size, VocabPin::Equals(_))
        && matches!(
            (config.architectures.as_slice(), id.architecture),
            ([architecture], Some(expected)) if architecture == expected
        )
        && config
            .dflash_config
            .as_ref()
            .is_some_and(|sub| sub.selector_vocab_size.is_some());
    usize::from(target_depth) + usize::from(draft_geometry) + usize::from(capture_taps) >= 2
        || selector_marker
}

/// Quorum rows that claim the checkpoint by its config alone.
pub(super) fn quorum_claimants(config: &DflashConfig) -> Vec<&'static DrafterProfile> {
    rows(DrafterAdmission::Quorum)
        .filter(|profile| quorum_claims(config, profile))
        .collect()
}

/// Quorum rows whose `fc.weight` shape `[H, H * taps]` the store carries.
pub(super) fn quorum_fc_claimants(fc_shape: &[usize]) -> Vec<&'static DrafterProfile> {
    rows(DrafterAdmission::Quorum)
        .filter(|profile| {
            let id = &profile.drafter;
            match (id.hidden_size, id.target_layer_ids) {
                (Some(hidden), Some(taps)) => fc_shape == [hidden, hidden * taps.len()],
                _ => false,
            }
        })
        .collect()
}

/// Resolve a claimed checkpoint to the one claiming row it matches exactly.
/// The architecture selects among rows that share an identity quorum.
pub(super) fn resolve_claimed(
    config: &DflashConfig,
    claimants: &[&'static DrafterProfile],
) -> Result<&'static DrafterProfile> {
    let candidates: Vec<_> = claimants
        .iter()
        .copied()
        .filter(|profile| architecture_matches(config, profile))
        .collect();
    if candidates.is_empty() {
        let labels: Vec<_> = claimants.iter().map(|p| p.label()).collect();
        let architectures: Vec<_> = claimants
            .iter()
            .filter_map(|p| p.drafter.architecture)
            .collect();
        bail!(
            "drafter config is claimed by profiles {labels:?}, whose architectures must be \
             exactly one of {architectures:?}; got {:?}",
            config.architectures
        );
    }
    first_match(config, &candidates)
}

/// The exact DFlash2 row a DFlash2-family checkpoint matches. Every
/// DFlash2-family checkpoint that no quorum row claims must match one.
pub(crate) fn exact_dflash2_profile(config: &DflashConfig) -> Result<&'static DrafterProfile> {
    let candidates: Vec<_> = rows(DrafterAdmission::Exact)
        .filter(|profile| profile.kind == DrafterKind::Dflash2)
        .collect();
    ensure!(
        !candidates.is_empty(),
        "no exact DFlash2 drafter profile is declared in any MODEL.toml; refusing the DFlash2 checkpoint"
    );
    first_match(config, &candidates)
}

fn first_match(
    config: &DflashConfig,
    candidates: &[&'static DrafterProfile],
) -> Result<&'static DrafterProfile> {
    let mut errors = Vec::new();
    for profile in candidates {
        match validate_identity(config, profile) {
            Ok(()) => return Ok(profile),
            Err(error) => errors.push(format!("{error:#}")),
        }
    }
    if let [only] = errors.as_slice() {
        bail!("{only}");
    }
    bail!(
        "drafter config matches none of the {} candidate profiles: {}",
        candidates.len(),
        errors.join("; ")
    )
}

fn architecture_matches(config: &DflashConfig, profile: &DrafterProfile) -> bool {
    match (
        config.architectures.as_slice(),
        profile.drafter.architecture,
    ) {
        ([architecture], Some(expected)) => architecture == expected,
        _ => false,
    }
}

/// Whether `config` satisfies every identity field the row pins.
pub(crate) fn matches_identity(config: &DflashConfig, profile: &DrafterProfile) -> bool {
    validate_identity(config, profile).is_ok()
}

/// Enforce every identity field the row pins, in a fixed order.
pub(super) fn validate_identity(config: &DflashConfig, profile: &DrafterProfile) -> Result<()> {
    let label = profile.label();
    let id = &profile.drafter;
    ensure!(
        matches!(
            profile.admission,
            DrafterAdmission::Exact | DrafterAdmission::Quorum
        ),
        "drafter profile {label} is {:?} and pins no identity",
        profile.admission
    );
    if let Some(expected) = id.architecture {
        ensure!(
            config.architectures.len() == 1 && config.architectures[0] == expected,
            "drafter profile {label} requires architectures=[{expected}]; got {:?}",
            config.architectures
        );
    }
    if let Some(expected) = id.model_type {
        ensure!(
            config.model_type.as_deref() == Some(expected),
            "drafter profile {label} model_type must be explicitly `{expected}`"
        );
    }
    let expected_family = match profile.kind {
        DrafterKind::Dspark => DrafterCheckpointFamily::Dspark,
        DrafterKind::Dflash | DrafterKind::Dflash2 => DrafterCheckpointFamily::Dflash,
        DrafterKind::Mtp => {
            bail!("drafter profile {label} is MTP and has no DFlash config identity")
        }
    };
    ensure!(
        config.checkpoint_family()? == expected_family,
        "drafter profile {label} must use the {expected_family} checkpoint family"
    );
    for (name, actual, expected) in [
        ("hidden_size", config.hidden_size, id.hidden_size),
        (
            "intermediate_size",
            config.intermediate_size,
            id.intermediate_size,
        ),
        (
            "num_hidden_layers",
            config.num_hidden_layers,
            id.num_hidden_layers,
        ),
        (
            "num_target_layers",
            config.num_target_layers,
            id.num_target_layers,
        ),
        (
            "num_attention_heads",
            config.num_attention_heads,
            id.num_attention_heads,
        ),
        (
            "num_key_value_heads",
            config.num_key_value_heads,
            id.num_key_value_heads,
        ),
        ("head_dim", config.head_dim, id.head_dim),
        ("vocab_size", config.vocab_size, id.vocab_size),
    ] {
        if let Some(expected) = expected {
            ensure!(
                actual == expected,
                "drafter profile {label} {name}={actual}; expected {expected}"
            );
        }
    }
    match id.draft_vocab_size {
        VocabPin::Unchecked => {}
        VocabPin::Undeclared => ensure!(
            config.draft_vocab_size.is_none(),
            "drafter profile {label} forbids a draft-to-target vocabulary remap"
        ),
        VocabPin::Equals(expected) => ensure!(
            config.draft_vocab_size.unwrap_or(config.vocab_size) == expected,
            "drafter profile {label} requires draft vocab {expected}"
        ),
    }
    let sub = config
        .dflash_config
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("drafter profile {label} requires dflash_config"))?;
    if profile.drafter.root_block_size_explicit {
        ensure!(
            config.root_block_size_explicit && Some(config.block_size) == id.block_size,
            "drafter profile {label} root block_size must be explicitly {:?}",
            id.block_size
        );
    }
    // The trained draft count is `block` for DSpark and `block - 1` for
    // DFlash/DFlash2. Not `resolve_draft_count`: its wide-width gate looks
    // this row up again.
    if let Some(block) = id.block_size {
        let expected_drafts = if profile.kind == DrafterKind::Dspark {
            block
        } else {
            block.saturating_sub(1)
        };
        ensure!(
            sub.block_size == Some(block) && config.resolved_block_size() == block,
            "drafter profile {label} requires nested block_size={block} (gamma{expected_drafts})"
        );
    }
    ensure!(
        !sub.fc_layernorm,
        "drafter profile {label} fc_layernorm must be false"
    );
    if let Some(mask) = id.mask_token_id {
        ensure!(
            sub.mask_token_id == mask && (mask as usize) < config.vocab_size,
            "drafter profile {label} requires mask_token_id={mask} within vocab"
        );
    }
    if let Some(taps) = id.target_layer_ids {
        ensure!(
            sub.target_layer_ids == taps,
            "drafter profile {label} requires ordered target_layer_ids {taps:?}"
        );
    }
    for (name, actual, expected) in [
        (
            "conv_kernel_size",
            sub.conv_kernel_size,
            id.conv_kernel_size,
        ),
        ("conv_group_size", sub.conv_group_size, id.conv_group_size),
        ("selector_rank", sub.selector_rank, id.selector_rank),
        ("selector_top_k", sub.selector_top_k, id.selector_top_k),
    ] {
        if let Some(expected) = expected {
            ensure!(
                actual == expected,
                "drafter profile {label} {name}={actual}; expected {expected}"
            );
        }
    }
    if let Some(group) = id.conv_group_size.filter(|group| *group > 0) {
        ensure!(
            config.hidden_size.is_multiple_of(group),
            "drafter profile {label} conv_group_size={group} must divide hidden"
        );
    }
    match id.selector_vocab_size {
        VocabPin::Unchecked => {}
        VocabPin::Undeclared => ensure!(
            sub.selector_vocab_size.is_none(),
            "drafter profile {label} forbids selector_vocab_size"
        ),
        VocabPin::Equals(expected) => ensure!(
            sub.selector_vocab_size == Some(expected),
            "drafter profile {label} selector_vocab_size must be explicitly {expected}"
        ),
    }
    if id.forbid_unknown_fields {
        ensure!(
            sub.unknown_fields.is_empty(),
            "drafter profile {label} has unknown dflash_config fields: {:?}",
            sub.unknown_fields.keys().collect::<Vec<_>>()
        );
    }
    if let Some(expected) = id.layer_types {
        ensure!(
            config.layer_types.as_ref().is_some_and(|layer_types| {
                layer_types
                    .iter()
                    .map(String::as_str)
                    .eq(expected.iter().copied())
            }),
            "drafter profile {label} layer_types must be exactly {expected:?}"
        );
    }
    if let Some(expected) = id.sliding_window {
        ensure!(
            config.sliding_window == Some(expected),
            "drafter profile {label} sliding_window must be explicitly {expected}"
        );
    }
    if let Some(expected) = id.is_causal {
        ensure!(
            config.is_causal == Some(expected),
            "drafter profile {label} is_causal must be explicitly {expected}"
        );
    }
    if let Some(expected) = id.tie_word_embeddings {
        ensure!(
            config.tie_word_embeddings == expected,
            "drafter profile {label} tie_word_embeddings must be {expected}"
        );
    }
    if let Some(theta) = id.rope_theta {
        let rope = config
            .rope_scaling
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("drafter profile {label} requires rope_parameters"))?;
        ensure!(
            config.rope_theta.to_bits() == theta.to_bits()
                && rope.rope_type.as_deref() == Some("default")
                && rope.rope_theta.map(f32::to_bits) == Some(theta.to_bits())
                && rope.factor.is_none()
                && rope.beta_fast.is_none()
                && rope.beta_slow.is_none()
                && rope.original_max_position_embeddings.is_none()
                && rope.attention_factor.is_none(),
            "drafter profile {label} requires default RoPE with theta={theta} and no scaling"
        );
    }
    if let Some(eps) = id.rms_norm_eps {
        ensure!(
            config.rms_norm_eps.map(f32::to_bits) == Some(eps.to_bits()),
            "drafter profile {label} requires explicit rms_norm_eps={eps}"
        );
    }
    if let Some(act) = id.hidden_act {
        ensure!(
            config.hidden_act.as_deref() == Some(act),
            "drafter profile {label} requires hidden_act={act}"
        );
    }
    // Enforce projector, Markov, and confidence-head policy from the matched
    // profile row instead of hardcoding DFlash/zero-Markov across all rows.
    let projector_ok = if profile.kind == DrafterKind::Dspark {
        sub.projector_type.is_none() || sub.projector_type.as_deref() == Some("dspark")
    } else {
        sub.projector_type.is_none()
    };
    let markov_ok = profile.allow_markov || config.markov_rank == 0;
    ensure!(
        projector_ok && markov_ok && config.confidence_head_config()?.is_none(),
        "drafter profile {label} forbids unsupported projector/Markov/confidence semantics"
    );
    Ok(())
}
