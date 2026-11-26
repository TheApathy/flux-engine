// SPDX-License-Identifier: AGPL-3.0-only

//! The `[[dflash.profile]]` table must reproduce the validators it replaced.
//!
//! `legacy` below is the pre-table code verbatim (constants and checks from
//! `dflash_validation/config.rs` and `native_flash_next.rs` at f95594dcc).
//! The tests prove two things: the resolved rows equal those constants field
//! by field, and over a sweep of single- and multi-field mutations the table
//! accepts and refuses exactly what the legacy validators did.

use atlas_kernels::{
    DrafterAdmission, DrafterCaptureMode, DrafterIdentity, DrafterKind, DrafterProfile,
    TargetIdentity, VocabPin,
};
use serde_json::json;

use super::config::validate_config;
use super::native_flash_next;
use super::profile::{self, validate_identity};
use super::qwen38_dflash2_tests::official_config;
use crate::weight_loader::dflash_loader::{DflashConfig, DflashRopeScaling, parse_dflash_config};

mod legacy {
    use anyhow::{Result, ensure};

    use crate::weight_loader::dflash_loader::{
        DflashConfig, DflashSubConfig, DrafterCheckpointFamily,
    };

    pub const HIDDEN: usize = 2_560;
    pub const INTERMEDIATE: usize = 8_704;
    pub const LAYERS: usize = 6;
    pub const TARGET_LAYERS: usize = 48;
    pub const QUERY_HEADS: usize = 20;
    pub const KV_HEADS: usize = 4;
    pub const HEAD_DIM: usize = 128;
    pub const VOCAB: usize = 248_320;
    pub const MASK_TOKEN: u32 = 248_077;
    pub const BLOCK: usize = 16;
    pub const DRAFTS: usize = 15;
    pub const V3_TENSOR_COUNT: usize = 69;
    pub const V3_PARAMETER_COUNT: usize = 547_918_336;
    pub const DFLASH2_TENSOR_COUNT: usize = 96;
    pub const DFLASH2_PARAMETER_COUNT: usize = 695_497_216;
    pub const CONV_KERNEL: usize = 2;
    pub const CONV_GROUP: usize = 16;
    pub const SELECTOR_RANK: usize = 256;
    pub const SELECTOR_TOP_K: usize = 16;
    pub const SELECTOR_VOCAB: usize = 248_077;
    pub const SLIDING_WINDOW: usize = 4_096;
    pub const TAPS: [usize; 8] = [1, 7, 13, 20, 26, 33, 39, 46];
    pub const DFLASH2_LAYER_TYPES: [&str; LAYERS] = [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ];
    // factory/build.rs
    pub const TARGET_VOCABS: [usize; 2] = [248_320, 248_077];
    // dflash_validation/qwen38_dflash2.rs
    pub const OFFICIAL_TENSOR_COUNT: usize = 81;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum NativeKind {
        V3,
        Dflash2,
    }

    pub fn looks_like_native_config(config: &DflashConfig) -> bool {
        let target_depth = config.num_target_layers == TARGET_LAYERS;
        let draft_geometry = config.hidden_size == HIDDEN
            && config.intermediate_size == INTERMEDIATE
            && config.num_hidden_layers == LAYERS;
        let capture_taps = config
            .dflash_config
            .as_ref()
            .is_some_and(|sub| sub.target_layer_ids == TAPS);
        let dflash2_selector_marker = matches!(
            config.architectures.as_slice(),
            [architecture] if architecture == "DFlash2DraftModel"
        ) && config
            .dflash_config
            .as_ref()
            .is_some_and(|sub| sub.selector_vocab_size.is_some());
        usize::from(target_depth) + usize::from(draft_geometry) + usize::from(capture_taps) >= 2
            || dflash2_selector_marker
    }

    pub fn validate_native_config(config: &DflashConfig) -> Result<NativeKind> {
        let kind = match config.architectures.as_slice() {
            [architecture] if architecture == "DFlashDraftModel" => NativeKind::V3,
            [architecture] if architecture == "DFlash2DraftModel" => NativeKind::Dflash2,
            _ => anyhow::bail!("native architectures"),
        };
        ensure!(config.model_type.as_deref() == Some("qwen3"));
        ensure!(config.checkpoint_family()? == DrafterCheckpointFamily::Dflash);
        for (actual, expected) in [
            (config.hidden_size, HIDDEN),
            (config.intermediate_size, INTERMEDIATE),
            (config.num_hidden_layers, LAYERS),
            (config.num_target_layers, TARGET_LAYERS),
            (config.num_attention_heads, QUERY_HEADS),
            (config.num_key_value_heads, KV_HEADS),
            (config.head_dim, HEAD_DIM),
            (config.vocab_size, VOCAB),
        ] {
            ensure!(actual == expected);
        }
        ensure!(config.root_block_size_explicit && config.block_size == BLOCK);
        let sub = config
            .dflash_config
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("dflash_config"))?;
        ensure!(sub.block_size == Some(BLOCK));
        ensure!(
            config.resolved_block_size() == BLOCK && config.resolve_draft_count(None)? == DRAFTS
        );
        ensure!(config.is_causal == Some(false));
        ensure!(!sub.fc_layernorm);
        ensure!(sub.mask_token_id == MASK_TOKEN);
        ensure!(sub.target_layer_ids == TAPS);
        ensure!(sub.projector_type.is_none());
        ensure!(config.markov_rank == 0 && config.confidence_head_config()?.is_none());
        ensure!(config.draft_vocab_size.is_none());
        ensure!(!config.tie_word_embeddings);
        match kind {
            NativeKind::V3 => ensure!(
                sub.conv_kernel_size == 0
                    && sub.conv_group_size == 0
                    && sub.selector_rank == 0
                    && sub.selector_top_k == 0
                    && sub.selector_vocab_size.is_none()
            ),
            NativeKind::Dflash2 => {
                ensure!(sub.conv_kernel_size == CONV_KERNEL);
                ensure!(sub.conv_group_size == CONV_GROUP);
                ensure!(sub.selector_rank == SELECTOR_RANK);
                ensure!(sub.selector_top_k == SELECTOR_TOP_K);
                ensure!(sub.selector_vocab_size == Some(SELECTOR_VOCAB));
                ensure!(sub.unknown_fields.is_empty());
                ensure!(config.sliding_window == Some(SLIDING_WINDOW));
                ensure!(config.layer_types.as_ref().is_some_and(|layer_types| {
                    layer_types
                        .iter()
                        .map(String::as_str)
                        .eq(DFLASH2_LAYER_TYPES)
                }));
            }
        }
        Ok(kind)
    }

    pub fn validate_config_if_candidate(config: &DflashConfig) -> Result<()> {
        if looks_like_native_config(config) {
            validate_native_config(config)?;
        }
        Ok(())
    }

    pub fn validate_qwen38_dflash2_config(
        config: &DflashConfig,
        sub: &DflashSubConfig,
    ) -> Result<()> {
        ensure!(config.architectures.len() == 1 && config.architectures[0] == "DFlash2DraftModel");
        ensure!(
            (
                config.hidden_size,
                config.intermediate_size,
                config.num_hidden_layers
            ) == (5120, 17408, 5)
        );
        ensure!(
            (
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim
            ) == (32, 8, 128)
        );
        ensure!(
            config.vocab_size == 248320
                && config.draft_vocab_size.unwrap_or(config.vocab_size) == 248320
        );
        ensure!(sub.block_size == Some(8) && config.resolved_block_size() == 8);
        ensure!(sub.mask_token_id == 248070 && sub.mask_token_id < config.vocab_size as u32);
        ensure!(sub.target_layer_ids == [5, 19, 33, 47, 61]);
        ensure!(
            (sub.conv_kernel_size, sub.conv_group_size) == (2, 16)
                && config.hidden_size.is_multiple_of(sub.conv_group_size)
        );
        ensure!((sub.selector_rank, sub.selector_top_k) == (256, 16));
        let layer_types = config
            .layer_types
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("layer_types"))?;
        ensure!(
            layer_types.len() == 5
                && layer_types
                    .iter()
                    .all(|layer_type| layer_type == "sliding_attention")
        );
        ensure!(config.sliding_window == Some(2048) && config.is_causal == Some(false));
        let rope = config
            .rope_scaling
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("rope_parameters"))?;
        ensure!(
            config.rope_theta.to_bits() == 10_000_000.0f32.to_bits()
                && rope.rope_type.as_deref() == Some("default")
                && rope.rope_theta.map(f32::to_bits) == Some(10_000_000.0f32.to_bits())
                && rope.factor.is_none()
                && rope.beta_fast.is_none()
                && rope.beta_slow.is_none()
                && rope.original_max_position_embeddings.is_none()
                && rope.attention_factor.is_none()
        );
        ensure!(
            config.rms_norm_eps.map(f32::to_bits) == Some(1e-6f32.to_bits())
                && config.hidden_act.as_deref() == Some("silu")
        );
        ensure!(
            sub.projector_type.is_none()
                && config.markov_rank == 0
                && !config.enable_confidence_head.unwrap_or(false)
                && !config.confidence_head_with_markov.unwrap_or(false)
                && !sub.enable_confidence_head.unwrap_or(false)
                && !sub.confidence_head_with_markov.unwrap_or(false)
        );
        Ok(())
    }
}

fn row(target_model: &str, name: &str) -> &'static DrafterProfile {
    profile::profiles()
        .iter()
        .find(|p| p.target_model == target_model && p.name == name)
        .unwrap_or_else(|| panic!("no [[dflash.profile]] row {target_model}/{name}"))
}

fn native_identity(dflash2: bool) -> DrafterIdentity {
    use legacy::*;
    DrafterIdentity {
        architecture: Some(if dflash2 {
            "DFlash2DraftModel"
        } else {
            "DFlashDraftModel"
        }),
        model_type: Some("qwen3"),
        hidden_size: Some(HIDDEN),
        intermediate_size: Some(INTERMEDIATE),
        num_hidden_layers: Some(LAYERS),
        num_target_layers: Some(TARGET_LAYERS),
        num_attention_heads: Some(QUERY_HEADS),
        num_key_value_heads: Some(KV_HEADS),
        head_dim: Some(HEAD_DIM),
        vocab_size: Some(VOCAB),
        draft_vocab_size: VocabPin::Undeclared,
        block_size: Some(BLOCK),
        root_block_size_explicit: true,
        mask_token_id: Some(MASK_TOKEN),
        target_layer_ids: Some(&TAPS),
        conv_kernel_size: Some(if dflash2 { CONV_KERNEL } else { 0 }),
        conv_group_size: Some(if dflash2 { CONV_GROUP } else { 0 }),
        selector_rank: Some(if dflash2 { SELECTOR_RANK } else { 0 }),
        selector_top_k: Some(if dflash2 { SELECTOR_TOP_K } else { 0 }),
        selector_vocab_size: if dflash2 {
            VocabPin::Equals(SELECTOR_VOCAB)
        } else {
            VocabPin::Undeclared
        },
        forbid_unknown_fields: dflash2,
        layer_types: dflash2.then_some(&DFLASH2_LAYER_TYPES[..]),
        sliding_window: dflash2.then_some(SLIDING_WINDOW),
        is_causal: Some(false),
        tie_word_embeddings: Some(false),
        rope_theta: None,
        rms_norm_eps: None,
        hidden_act: None,
        tensor_count: Some(if dflash2 {
            DFLASH2_TENSOR_COUNT
        } else {
            V3_TENSOR_COUNT
        }),
        parameter_count: Some(if dflash2 {
            DFLASH2_PARAMETER_COUNT
        } else {
            V3_PARAMETER_COUNT
        }),
        bare_tensor_names: true,
    }
}

fn native_profile(name: &'static str, dflash2: bool) -> DrafterProfile {
    use legacy::*;
    DrafterProfile {
        target_model: "qwen3.8-flash-next",
        name,
        kind: if dflash2 {
            DrafterKind::Dflash2
        } else {
            DrafterKind::Dflash
        },
        admission: DrafterAdmission::Quorum,
        // Pairing keeps requiring ATLAS_EXPERIMENTAL_NATIVE_QWEN4_DFLASH=1.
        qualified: false,
        launcher_default: false,
        capture_mode: DrafterCaptureMode::Qwen4HyperProjected,
        verify_k: Some((BLOCK, BLOCK)),
        wide_verify_k: None,
        proposal_vocab: dflash2.then_some(SELECTOR_VOCAB),
        allow_modelopt_nvfp4: false,
        allow_rtncal_w4a4: false,
        allow_markov: false,
        target: TargetIdentity {
            hidden_size: Some(HIDDEN),
            residual_width: Some(4 * HIDDEN),
            num_layers: Some(TARGET_LAYERS),
            vocab_sizes: Some(&TARGET_VOCABS),
        },
        drafter: native_identity(dflash2),
    }
}

#[test]
fn flash_next_native_rows_equal_the_legacy_constants_field_by_field() {
    assert_eq!(
        *row("qwen3.8-flash-next", "native-v3"),
        native_profile("native-v3", false)
    );
    assert_eq!(
        *row("qwen3.8-flash-next", "native-dflash2"),
        native_profile("native-dflash2", true)
    );
    // DRAFTS is not a row field: it is the trained block minus the anchor.
    assert_eq!(legacy::DRAFTS, legacy::BLOCK - 1);
}

#[test]
fn qwen38_27b_dflash2_row_equals_the_legacy_strict_profile_field_by_field() {
    let expected = DrafterProfile {
        target_model: "qwen3.8-27b",
        name: "zlab-dflash2",
        kind: DrafterKind::Dflash2,
        admission: DrafterAdmission::Exact,
        qualified: true,
        launcher_default: true,
        capture_mode: DrafterCaptureMode::ResidualSlice,
        verify_k: Some((2, 8)),
        // `draft_width_is_supported`: trained 7 drafts widen only to 15.
        wide_verify_k: Some(16),
        proposal_vocab: None,
        allow_modelopt_nvfp4: true,
        allow_rtncal_w4a4: true,
        allow_markov: false,
        target: TargetIdentity {
            hidden_size: None,
            residual_width: None,
            num_layers: None,
            vocab_sizes: None,
        },
        drafter: DrafterIdentity {
            architecture: Some("DFlash2DraftModel"),
            model_type: None,
            hidden_size: Some(5120),
            intermediate_size: Some(17408),
            num_hidden_layers: Some(5),
            num_target_layers: None,
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            head_dim: Some(128),
            vocab_size: Some(248320),
            draft_vocab_size: VocabPin::Equals(248320),
            block_size: Some(8),
            root_block_size_explicit: false,
            mask_token_id: Some(248070),
            target_layer_ids: Some(&[5, 19, 33, 47, 61]),
            conv_kernel_size: Some(2),
            conv_group_size: Some(16),
            selector_rank: Some(256),
            selector_top_k: Some(16),
            selector_vocab_size: VocabPin::Unchecked,
            forbid_unknown_fields: false,
            layer_types: Some(&["sliding_attention"; 5]),
            sliding_window: Some(2048),
            is_causal: Some(false),
            tie_word_embeddings: None,
            rope_theta: Some(10_000_000.0),
            rms_norm_eps: Some(1e-6),
            hidden_act: Some("silu"),
            tensor_count: Some(legacy::OFFICIAL_TENSOR_COUNT),
            parameter_count: None,
            bare_tensor_names: false,
        },
    };
    let actual = row("qwen3.8-27b", "zlab-dflash2");
    assert_eq!(*actual, expected);
    // f32 equality above is IEEE equality; the legacy check compared bits.
    assert_eq!(
        actual.drafter.rope_theta.map(f32::to_bits),
        Some(10_000_000.0f32.to_bits())
    );
    assert_eq!(
        actual.drafter.rms_norm_eps.map(f32::to_bits),
        Some(1e-6f32.to_bits())
    );
}

#[test]
fn only_the_legacy_profiles_pin_an_identity() {
    // A new exact/quorum row widens admission; it must come with its own
    // equivalence or acceptance evidence, so this list changes deliberately.
    let mut pinned: Vec<_> = profile::profiles()
        .iter()
        .filter(|p| {
            matches!(
                p.admission,
                DrafterAdmission::Exact | DrafterAdmission::Quorum
            )
        })
        .map(DrafterProfile::label)
        .collect();
    pinned.sort();
    assert_eq!(
        pinned,
        [
            "qwen3.8-27b/zlab-dflash2",
            "qwen3.8-flash-next/incoai-dflash2-b8",
            "qwen3.8-flash-next/native-dflash2",
            "qwen3.8-flash-next/native-v3",
            "qwen3.8-flash-next/pixelml-qwen3dspark",
        ]
    );
}

#[test]
fn refusals_stay_out_of_the_table() {
    // GLM speculation stays refused; Gemma and DeepSeek-V4.1 have no
    // external drafter row (Gemma until its multi-row verify attention
    // exists, DeepSeek because DSpark is internal).
    for profile in profile::profiles() {
        if profile.target_model == "glm5.3-flash" {
            assert_eq!(profile.admission, DrafterAdmission::Refused, "{profile:?}");
        }
        assert!(
            !profile.target_model.starts_with("gemma")
                && !profile.target_model.starts_with("deepseek"),
            "{}",
            profile.label()
        );
        assert!(
            profile.capture_mode.is_implemented() || !profile.qualified,
            "{} is qualified on an unimplemented capture mode",
            profile.label()
        );
    }
}

type Mutation = (&'static str, fn(&mut DflashConfig));

fn sub(config: &mut DflashConfig) -> &mut crate::weight_loader::dflash_loader::DflashSubConfig {
    config
        .dflash_config
        .as_mut()
        .expect("fixture has dflash_config")
}

/// Mutations of every field either validator reads, plus multi-axis
/// mutations that move a config across the native quorum.
const MUTATIONS: &[Mutation] = &[
    ("identity", |_| {}),
    ("arch other", |c| c.architectures = vec!["Other".into()]),
    ("arch v3", |c| {
        c.architectures = vec!["DFlashDraftModel".into()]
    }),
    ("arch dflash2", |c| {
        c.architectures = vec!["DFlash2DraftModel".into()]
    }),
    ("arch dspark", |c| {
        c.architectures = vec!["DSparkDraftModel".into()]
    }),
    ("arch two", |c| c.architectures.push("Extra".into())),
    ("arch none", |c| c.architectures.clear()),
    ("model_type none", |c| c.model_type = None),
    ("model_type qwen4", |c| c.model_type = Some("qwen4".into())),
    ("model_type qwen3", |c| c.model_type = Some("qwen3".into())),
    ("hidden+1", |c| c.hidden_size += 1),
    ("hidden+16", |c| c.hidden_size += 16),
    ("inter+1", |c| c.intermediate_size += 1),
    ("layers+1", |c| c.num_hidden_layers += 1),
    ("layers-1", |c| c.num_hidden_layers -= 1),
    ("target layers+1", |c| c.num_target_layers += 1),
    ("target layers 48", |c| c.num_target_layers = 48),
    ("target layers 64", |c| c.num_target_layers = 64),
    ("heads+1", |c| c.num_attention_heads += 1),
    ("kv*2", |c| c.num_key_value_heads *= 2),
    ("head_dim 256", |c| c.head_dim = 256),
    ("vocab+1", |c| c.vocab_size += 1),
    ("draft vocab same", |c| {
        c.draft_vocab_size = Some(c.vocab_size)
    }),
    ("draft vocab small", |c| c.draft_vocab_size = Some(32_000)),
    ("tie", |c| c.tie_word_embeddings = true),
    ("root block 32", |c| c.block_size = 32),
    ("root block 8", |c| c.block_size = 8),
    ("root implicit", |c| c.root_block_size_explicit = false),
    ("nested block none", |c| sub(c).block_size = None),
    ("nested block 32", |c| sub(c).block_size = Some(32)),
    ("nested block 7", |c| sub(c).block_size = Some(7)),
    ("mask+1", |c| sub(c).mask_token_id += 1),
    ("mask past vocab", |c| sub(c).mask_token_id = 300_000),
    ("tap moved", |c| sub(c).target_layer_ids[1] += 1),
    ("taps reversed", |c| sub(c).target_layer_ids.reverse()),
    ("taps short", |c| {
        sub(c).target_layer_ids.pop();
    }),
    ("taps 27b", |c| {
        sub(c).target_layer_ids = vec![5, 19, 33, 47, 61]
    }),
    ("taps daocloud", |c| {
        sub(c).target_layer_ids = vec![6, 20, 34, 48, 62]
    }),
    ("taps native", |c| {
        sub(c).target_layer_ids = vec![1, 7, 13, 20, 26, 33, 39, 46]
    }),
    ("conv kernel 3", |c| sub(c).conv_kernel_size = 3),
    ("conv kernel 0", |c| sub(c).conv_kernel_size = 0),
    ("conv group 8", |c| sub(c).conv_group_size = 8),
    ("conv group 0", |c| sub(c).conv_group_size = 0),
    ("conv 2/16", |c| {
        sub(c).conv_kernel_size = 2;
        sub(c).conv_group_size = 16;
    }),
    ("selector rank 128", |c| sub(c).selector_rank = 128),
    ("selector top_k 8", |c| sub(c).selector_top_k = 8),
    ("selector 256/16", |c| {
        sub(c).selector_rank = 256;
        sub(c).selector_top_k = 16;
    }),
    ("selector vocab none", |c| sub(c).selector_vocab_size = None),
    ("selector vocab native", |c| {
        sub(c).selector_vocab_size = Some(248_077)
    }),
    ("selector vocab physical", |c| {
        sub(c).selector_vocab_size = Some(248_320)
    }),
    ("unknown nested field", |c| {
        sub(c).unknown_fields.insert("future".into(), json!(1));
    }),
    ("layer types none", |c| c.layer_types = None),
    ("layer types all full", |c| {
        if let Some(types) = c.layer_types.as_mut() {
            types.iter_mut().for_each(|t| *t = "full_attention".into());
        }
    }),
    ("layer types native", |c| {
        c.layer_types = Some(legacy::DFLASH2_LAYER_TYPES.map(String::from).to_vec())
    }),
    ("layer types 5 sliding", |c| {
        c.layer_types = Some(vec!["sliding_attention".into(); 5])
    }),
    ("swa none", |c| c.sliding_window = None),
    ("swa 2048", |c| c.sliding_window = Some(2048)),
    ("swa 4096", |c| c.sliding_window = Some(4096)),
    ("causal none", |c| c.is_causal = None),
    ("causal true", |c| c.is_causal = Some(true)),
    ("fc_layernorm", |c| sub(c).fc_layernorm = true),
    ("projector dspark", |c| {
        sub(c).projector_type = Some("dspark".into())
    }),
    ("projector other", |c| {
        sub(c).projector_type = Some("mlp".into())
    }),
    ("markov", |c| c.markov_rank = 256),
    ("confidence root", |c| c.enable_confidence_head = Some(true)),
    ("confidence root false", |c| {
        c.enable_confidence_head = Some(false)
    }),
    ("confidence nested", |c| {
        sub(c).enable_confidence_head = Some(true)
    }),
    ("confidence disagree", |c| {
        c.enable_confidence_head = Some(false);
        sub(c).enable_confidence_head = Some(true);
    }),
    ("with_markov root", |c| {
        c.confidence_head_with_markov = Some(true)
    }),
    ("with_markov root false", |c| {
        c.confidence_head_with_markov = Some(false)
    }),
    ("with_markov nested", |c| {
        sub(c).confidence_head_with_markov = Some(true)
    }),
    ("rope theta", |c| c.rope_theta = 1_000_000.0),
    ("rope none", |c| c.rope_scaling = None),
    ("rope yarn", |c| {
        if let Some(rope) = c.rope_scaling.as_mut() {
            rope.rope_type = Some("yarn".into());
        }
    }),
    ("rope factor", |c| {
        if let Some(rope) = c.rope_scaling.as_mut() {
            rope.factor = Some(4.0);
        }
    }),
    ("rope inner theta", |c| {
        if let Some(rope) = c.rope_scaling.as_mut() {
            rope.rope_theta = Some(1_000_000.0);
        }
    }),
    ("rope default", |c| {
        c.rope_scaling = Some(DflashRopeScaling {
            rope_type: Some("default".into()),
            factor: None,
            beta_fast: None,
            beta_slow: None,
            original_max_position_embeddings: None,
            rope_theta: Some(c.rope_theta),
            attention_factor: None,
        })
    }),
    ("eps none", |c| c.rms_norm_eps = None),
    ("eps 1e-5", |c| c.rms_norm_eps = Some(1e-5)),
    ("act none", |c| c.hidden_act = None),
    ("act gelu", |c| c.hidden_act = Some("gelu".into())),
    ("quorum: depth+geometry off", |c| {
        c.num_target_layers += 1;
        c.hidden_size += 16;
    }),
    ("quorum: geometry+taps off", |c| {
        c.intermediate_size += 1;
        sub(c).target_layer_ids[0] += 1;
    }),
    ("quorum: all three off", |c| {
        c.num_target_layers += 1;
        c.num_hidden_layers += 1;
        sub(c).target_layer_ids[0] += 1;
    }),
    ("27b geometry", |c| {
        c.hidden_size = 5120;
        c.intermediate_size = 17408;
        c.num_hidden_layers = 5;
    }),
    ("native geometry", |c| {
        c.hidden_size = 2560;
        c.intermediate_size = 8704;
        c.num_hidden_layers = 6;
    }),
];

fn native_v3() -> DflashConfig {
    let config = json!({
        "architectures": ["DFlashDraftModel"],
        "block_size": 16,
        "dflash_config": {
            "block_size": 16,
            "mask_token_id": 248077,
            "target_layer_ids": [1, 7, 13, 20, 26, 33, 39, 46]
        },
        "head_dim": 128,
        "hidden_size": 2560,
        "intermediate_size": 8704,
        "is_causal": false,
        "model_type": "qwen3",
        "num_attention_heads": 20,
        "num_hidden_layers": 6,
        "num_key_value_heads": 4,
        "num_target_layers": 48,
        "tie_word_embeddings": false,
        "vocab_size": 248320
    });
    parse_dflash_config(&config.to_string()).expect("native V3 fixture")
}

fn native_dflash2() -> DflashConfig {
    let mut config = native_v3();
    config.architectures = vec!["DFlash2DraftModel".into()];
    let nested = sub(&mut config);
    nested.conv_kernel_size = 2;
    nested.conv_group_size = 16;
    nested.selector_rank = 256;
    nested.selector_top_k = 16;
    nested.selector_vocab_size = Some(248_077);
    config.layer_types = Some(legacy::DFLASH2_LAYER_TYPES.map(String::from).to_vec());
    config.sliding_window = Some(4096);
    config
}

fn qwen38_27b() -> DflashConfig {
    official_config()
}

fn fixtures() -> [(&'static str, DflashConfig); 3] {
    [
        ("native-v3", native_v3()),
        ("native-dflash2", native_dflash2()),
        ("27b-dflash2", qwen38_27b()),
    ]
}

#[test]
fn quorum_claims_exactly_what_looks_like_native_config_claimed() {
    for (fixture, base) in fixtures() {
        for (name, mutate) in MUTATIONS {
            let mut config = base.clone();
            mutate(&mut config);
            assert_eq!(
                !native_flash_next::quorum_claimants(&config).is_empty(),
                legacy::looks_like_native_config(&config),
                "{fixture} / {name}"
            );
        }
    }
}

#[test]
fn native_config_admission_matches_the_legacy_validator() {
    let (mut claimed_ok, mut claimed_refused) = (0, 0);
    for (fixture, base) in fixtures() {
        for (name, mutate) in MUTATIONS {
            let mut config = base.clone();
            mutate(&mut config);
            let legacy = legacy::validate_config_if_candidate(&config);
            let table = native_flash_next::validate_config_if_candidate(&config);
            assert_eq!(
                table.is_ok(),
                legacy.is_ok(),
                "{fixture} / {name}: table={table:?} legacy={legacy:?}"
            );
            if legacy::looks_like_native_config(&config) {
                if legacy.is_ok() {
                    claimed_ok += 1;
                } else {
                    claimed_refused += 1;
                }
            }
        }
    }
    // The sweep must exercise both outcomes of the claimed path.
    assert!(
        claimed_ok >= 10 && claimed_refused >= 40,
        "{claimed_ok}/{claimed_refused}"
    );
}

#[test]
fn native_row_selection_matches_the_legacy_kind() {
    for (fixture, base) in fixtures() {
        for (name, mutate) in MUTATIONS {
            let mut config = base.clone();
            mutate(&mut config);
            let Ok(kind) = legacy::validate_native_config(&config) else {
                continue;
            };
            let claimants = native_flash_next::quorum_claimants(&config);
            let resolved = profile::resolve_claimed(&config, &claimants)
                .unwrap_or_else(|e| panic!("{fixture} / {name}: {e:#}"));
            let expected = match kind {
                legacy::NativeKind::V3 => "native-v3",
                legacy::NativeKind::Dflash2 => "native-dflash2",
            };
            assert_eq!(resolved.name, expected, "{fixture} / {name}");
        }
    }
}

#[test]
fn exact_dflash2_admission_matches_the_legacy_strict_profile() {
    let zlab = row("qwen3.8-27b", "zlab-dflash2");
    let (mut accepted, mut refused) = (0, 0);
    for (fixture, base) in fixtures() {
        for (name, mutate) in MUTATIONS {
            let mut config = base.clone();
            mutate(&mut config);
            // `validate_config` runs the generic checks the strict profile
            // always sat behind; compare only where they pass.
            if validate_config(&config, false).is_err() {
                continue;
            }
            let Some(nested) = config.dflash_config.clone() else {
                continue;
            };
            let legacy = legacy::validate_qwen38_dflash2_config(&config, &nested);
            if legacy.is_ok() {
                accepted += 1;
            } else {
                refused += 1;
            }
            let table = validate_identity(&config, zlab);
            assert_eq!(
                table.is_ok(),
                legacy.is_ok(),
                "{fixture} / {name}: table={table:?} legacy={legacy:?}"
            );
            assert_eq!(
                super::exact_dflash2_profile(&config).is_ok(),
                super::config::declares_dflash2_family(&config) && legacy.is_ok(),
                "{fixture} / {name}"
            );
        }
    }
    assert!(accepted >= 5 && refused >= 60, "{accepted}/{refused}");
}

#[test]
fn every_fixture_is_admitted_by_its_own_row_only() {
    assert_eq!(
        profile::resolve_claimed(
            &native_v3(),
            &native_flash_next::quorum_claimants(&native_v3())
        )
        .unwrap()
        .name,
        "native-v3"
    );
    let dflash2 = native_dflash2();
    assert_eq!(
        profile::resolve_claimed(&dflash2, &native_flash_next::quorum_claimants(&dflash2))
            .unwrap()
            .name,
        "native-dflash2"
    );
    assert_eq!(
        super::exact_dflash2_profile(&qwen38_27b()).unwrap().name,
        "zlab-dflash2"
    );
    assert!(super::exact_dflash2_profile(&dflash2).is_err());
    assert!(native_flash_next::quorum_claimants(&qwen38_27b()).is_empty());
}

const PIXELML_QWEN3DSPARK_JSON: &str = r#"{
  "architectures": ["Qwen3DSparkModel"],
  "attention_bias": false,
  "block_size": 7,
  "dtype": "bfloat16",
  "enable_confidence_head": false,
  "head_dim": 256,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "intermediate_size": 7680,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "markov_rank": 0,
  "mask_token_id": 248077,
  "max_position_embeddings": 262144,
  "model_type": "qwen3",
  "num_attention_heads": 24,
  "num_hidden_layers": 5,
  "num_key_value_heads": 2,
  "num_target_layers": 48,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
    "rope_theta": 10000000.0,
    "rope_type": "default"
  },
  "sliding_window": null,
  "target_layer_ids": [3, 15, 23, 35, 43],
  "target_model_type": "qwen4_exp",
  "tie_word_embeddings": false,
  "vocab_size": 248320
}"#;

#[test]
fn pixelml_qwen3dspark_config_and_58_tensor_store_admit_cleanly() {
    use std::collections::HashMap;

    use spark_runtime::weights::WeightDtype;

    use super::qwen38_dflash2_tests::FakeStore;
    use super::validate_dflash_metadata;
    use crate::weight_loader::dflash_loader::DrafterCheckpointFamily;

    let config = parse_dflash_config(PIXELML_QWEN3DSPARK_JSON).expect("parse PixelML config");
    assert_eq!(
        config.checkpoint_family().expect("family"),
        DrafterCheckpointFamily::Dspark
    );
    assert_eq!(config.markov_rank, 0);
    assert_eq!(config.target_model_type.as_deref(), Some("qwen4_exp"));
    let normalized = config.dflash_config.as_ref().expect("normalized subconfig");
    assert_eq!(normalized.mask_token_id, 248_077);
    assert_eq!(normalized.target_layer_ids, vec![3, 15, 23, 35, 43]);
    assert_eq!(config.resolve_draft_count(None).expect("drafts"), 7);

    let claimants = native_flash_next::quorum_claimants(&config);
    let resolved = profile::resolve_claimed(&config, &claimants).expect("pixelml profile");
    assert_eq!(resolved.label(), "qwen3.8-flash-next/pixelml-qwen3dspark");
    assert_eq!(resolved.capture_mode, DrafterCaptureMode::Qwen4LayerAttnHc);
    assert_eq!(resolved.verify_k, Some((2, 8)));

    let manifest = native_flash_next::finiteness_manifest_if_candidate(&config)
        .expect("manifest result")
        .expect("pixelml manifest");
    assert_eq!(manifest.len(), 58);
    let total_params: usize = manifest
        .values()
        .map(|shape| shape.iter().product::<usize>())
        .sum();
    assert_eq!(total_params, 498_106_880);

    let mut tensors = HashMap::new();
    for (name, shape) in manifest {
        tensors.insert(name, (shape, WeightDtype::BF16));
    }
    let store = FakeStore(tensors);
    assert_eq!(
        validate_dflash_metadata(&store, &config).expect("58-tensor store admission"),
        Some("")
    );

    // Top-level vs nested disagreement must fail closed.
    let mut conflicting: serde_json::Value =
        serde_json::from_str(PIXELML_QWEN3DSPARK_JSON).expect("valid json");
    conflicting["dflash_config"] = json!({
        "block_size": 7,
        "mask_token_id": 248070,
        "target_layer_ids": [3, 15, 23, 35, 43]
    });
    assert!(parse_dflash_config(&conflicting.to_string()).is_err());
}

#[test]
fn synthetic_incoai_b8_markov64_admits_via_qwen4_hc_mean_profile() {
    use std::collections::HashMap;

    use spark_runtime::weights::WeightDtype;

    use super::qwen38_dflash2_tests::FakeStore;
    use super::validate_dflash_metadata;

    let incoai_json = json!({
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_size": 2560,
        "intermediate_size": 7680,
        "num_hidden_layers": 5,
        "num_target_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 248320,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention"
        ],
        "sliding_window": 2048,
        "is_causal": false,
        "tie_word_embeddings": false,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "markov_rank": 64,
        "markov_head_type": "vanilla",
        "dflash_config": {
            "block_size": 8,
            "mask_token_id": 248077,
            "target_layer_ids": [5, 15, 25, 35, 45],
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "selector_rank": 256,
            "selector_top_k": 16
        }
    });
    let config = parse_dflash_config(&incoai_json.to_string()).expect("parse incoai config");
    assert_eq!(config.resolved_block_size(), 8);
    assert_eq!(config.markov_rank, 64);

    let claimants = native_flash_next::quorum_claimants(&config);
    let resolved = profile::resolve_claimed(&config, &claimants).expect("incoai profile");
    assert_eq!(resolved.label(), "qwen3.8-flash-next/incoai-dflash2-b8");
    assert_eq!(resolved.capture_mode, DrafterCaptureMode::Qwen4HcMean);
    assert_eq!(resolved.verify_k, Some((2, 9)));
    assert!(resolved.allow_markov);

    let manifest = native_flash_next::finiteness_manifest_if_candidate(&config)
        .expect("manifest result")
        .expect("incoai manifest");
    assert_eq!(manifest.len(), 83);
    assert!(manifest.contains_key("markov_head.markov_w1.weight"));
    assert!(manifest.contains_key("markov_head.markov_w2.weight"));

    let mut tensors = HashMap::new();
    for (name, shape) in manifest {
        tensors.insert(name, (shape, WeightDtype::BF16));
    }
    let store = FakeStore(tensors);
    assert_eq!(
        validate_dflash_metadata(&store, &config).expect("incoai markov64 store admission"),
        Some("")
    );
}

#[test]
fn released_incoai_flash_next_dflash2_config_and_81_tensor_store_admit_cleanly() {
    use std::collections::HashMap;

    use spark_runtime::weights::WeightDtype;

    use super::qwen38_dflash2_tests::FakeStore;
    use super::validate_dflash_metadata;

    let released_json = json!({
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_size": 2560,
        "intermediate_size": 7680,
        "num_hidden_layers": 5,
        "num_target_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 248320,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention"
        ],
        "sliding_window": 2048,
        "is_causal": false,
        "tie_word_embeddings": false,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "rope_parameters": {
            "rope_theta": 10000,
            "rope_type": "default"
        },
        "dflash_config": {
            "block_size": 8,
            "mask_token_id": 248077,
            "target_layer_ids": [5, 15, 25, 35, 45],
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "selector_rank": 256,
            "selector_top_k": 16
        }
    });
    let config =
        parse_dflash_config(&released_json.to_string()).expect("parse released incoai config");
    assert_eq!(config.resolved_block_size(), 8);
    assert_eq!(config.resolve_draft_count(None).expect("drafts"), 7);
    assert_eq!(config.markov_rank, 0);
    assert_eq!(
        config.rope_scaling.as_ref().and_then(|r| r.rope_theta),
        Some(10_000.0)
    );

    let claimants = native_flash_next::quorum_claimants(&config);
    let resolved = profile::resolve_claimed(&config, &claimants).expect("incoai profile");
    assert_eq!(resolved.label(), "qwen3.8-flash-next/incoai-dflash2-b8");
    assert_eq!(resolved.capture_mode, DrafterCaptureMode::Qwen4HcMean);
    assert_eq!(resolved.verify_k, Some((2, 9)));

    let manifest = native_flash_next::finiteness_manifest_if_candidate(&config)
        .expect("manifest result")
        .expect("incoai manifest");
    assert_eq!(manifest.len(), 81);
    assert!(!manifest.contains_key("markov_head.markov_w1.weight"));
    let total_params: usize = manifest
        .values()
        .map(|shape| shape.iter().product::<usize>())
        .sum();
    assert_eq!(total_params, 603_065_600);

    let mut tensors = HashMap::new();
    for (name, shape) in manifest {
        tensors.insert(name, (shape, WeightDtype::BF16));
    }
    let store = FakeStore(tensors);
    assert_eq!(
        validate_dflash_metadata(&store, &config).expect("81-tensor incoai store admission"),
        Some("")
    );

    // Mismatched head geometry (legacy 20Q:4KV) must fail closed.
    let mut wrong_heads = config;
    wrong_heads.num_attention_heads = 20;
    wrong_heads.num_key_value_heads = 4;
    let wrong_claimants = native_flash_next::quorum_claimants(&wrong_heads);
    assert!(profile::resolve_claimed(&wrong_heads, &wrong_claimants).is_err());
}
