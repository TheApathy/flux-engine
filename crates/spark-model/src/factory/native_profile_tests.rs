// SPDX-License-Identifier: AGPL-3.0-only

//! The factory's native Flash-Next pairing now reads `[[dflash.profile]]`
//! rows. `legacy_is_native` is the pre-table check verbatim (f95594dcc);
//! the sweep proves the row lookup pairs exactly the same target/drafter
//! combinations and resolves the same capture mode, width and vocabulary.

use atlas_core::config::{DflashCaptureMode, ModelConfig};

use super::{
    is_native_qwen38_flash_next_drafter, native_qwen38_flash_next_profile,
    resolve_native_dflash2_proposal_vocab, runtime_capture_mode, validate_dflash_capture_residual,
    validate_native_qwen4_dflash_width,
};
use crate::weight_loader::DflashConfig;

const QWEN38_FLASH_NEXT_HIDDEN: usize = 2560;
const QWEN38_FLASH_NEXT_VOCAB: usize = 248_320;
const QWEN38_FLASH_NEXT_TARGET_LAYERS: usize = 48;
const QWEN38_FLASH_NEXT_DRAFT_LAYERS: usize = 6;
const QWEN38_FLASH_NEXT_DRAFT_BLOCK_SIZE: usize = 16;
const QWEN38_FLASH_NEXT_DRAFT_GAMMA: usize = QWEN38_FLASH_NEXT_DRAFT_BLOCK_SIZE - 1;
const QWEN38_FLASH_NEXT_CAPTURE_LAYERS: [usize; 8] = [1, 7, 13, 20, 26, 33, 39, 46];
const QWEN38_FLASH_NEXT_SELECTOR_VOCAB: usize = 248_077;
const QWEN38_FLASH_NEXT_DFLASH2_LAYER_TYPES: [&str; QWEN38_FLASH_NEXT_DRAFT_LAYERS] = [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
];

fn legacy_is_native(target: &ModelConfig, drafter: &DflashConfig) -> bool {
    let Some(sub) = drafter.dflash_config.as_ref() else {
        return false;
    };
    let common = target.is_qwen4_exp()
        && target.hidden_size == QWEN38_FLASH_NEXT_HIDDEN
        && target.residual_width() == 4 * QWEN38_FLASH_NEXT_HIDDEN
        && matches!(
            target.vocab_size,
            QWEN38_FLASH_NEXT_VOCAB | QWEN38_FLASH_NEXT_SELECTOR_VOCAB
        )
        && target.num_hidden_layers == QWEN38_FLASH_NEXT_TARGET_LAYERS
        && drafter.hidden_size == QWEN38_FLASH_NEXT_HIDDEN
        && drafter.vocab_size == QWEN38_FLASH_NEXT_VOCAB
        && drafter.num_hidden_layers == QWEN38_FLASH_NEXT_DRAFT_LAYERS
        && drafter.num_target_layers == QWEN38_FLASH_NEXT_TARGET_LAYERS
        && drafter.intermediate_size == 8704
        && drafter.num_attention_heads == 20
        && drafter.num_key_value_heads == 4
        && drafter.head_dim == 128
        && drafter.model_type.as_deref() == Some("qwen3")
        && drafter.root_block_size_explicit
        && drafter.block_size == QWEN38_FLASH_NEXT_DRAFT_BLOCK_SIZE
        && drafter.resolved_block_size() == QWEN38_FLASH_NEXT_DRAFT_BLOCK_SIZE
        && drafter.is_causal == Some(false)
        && !drafter.tie_word_embeddings
        && drafter.draft_vocab_size.is_none()
        && drafter.markov_rank == 0
        && drafter
            .confidence_head_config()
            .is_ok_and(|confidence| confidence.is_none())
        && sub.block_size == Some(QWEN38_FLASH_NEXT_DRAFT_BLOCK_SIZE)
        && sub.mask_token_id == 248_077
        && sub.target_layer_ids == QWEN38_FLASH_NEXT_CAPTURE_LAYERS
        && sub.projector_type.is_none()
        && !sub.fc_layernorm;
    if !common {
        return false;
    }
    match drafter.architectures.as_slice() {
        [architecture] if architecture == "DFlashDraftModel" => {
            sub.conv_kernel_size == 0
                && sub.conv_group_size == 0
                && sub.selector_rank == 0
                && sub.selector_top_k == 0
                && sub.selector_vocab_size.is_none()
        }
        [architecture] if architecture == "DFlash2DraftModel" => {
            sub.conv_kernel_size == 2
                && sub.conv_group_size == 16
                && sub.selector_rank == 256
                && sub.selector_top_k == 16
                && sub.selector_vocab_size == Some(QWEN38_FLASH_NEXT_SELECTOR_VOCAB)
                && drafter.sliding_window == Some(4_096)
                && drafter.layer_types.as_ref().is_some_and(|layer_types| {
                    layer_types
                        .iter()
                        .map(String::as_str)
                        .eq(QWEN38_FLASH_NEXT_DFLASH2_LAYER_TYPES)
                })
        }
        _ => false,
    }
}

fn flash_next_target() -> ModelConfig {
    let mut target = ModelConfig::qwen3_next_80b_nvfp4();
    target.model_type = "qwen4_exp".to_string();
    target.hidden_size = 2560;
    target.hc_count = 4;
    target.vocab_size = 248_320;
    target.num_hidden_layers = 48;
    target
}

fn native_v3() -> DflashConfig {
    crate::weight_loader::dflash_loader::parse_dflash_config(
        r#"{
            "architectures":["DFlashDraftModel"], "model_type":"qwen3",
            "hidden_size":2560, "num_hidden_layers":6, "num_target_layers":48,
            "intermediate_size":8704, "num_attention_heads":20, "num_key_value_heads":4,
            "head_dim":128, "vocab_size":248320, "block_size":16,
            "tie_word_embeddings":false, "is_causal":false,
            "dflash_config":{"block_size":16, "mask_token_id":248077,
                             "target_layer_ids":[1,7,13,20,26,33,39,46]}
        }"#,
    )
    .expect("native V3 fixture")
}

fn native_dflash2() -> DflashConfig {
    let mut drafter = native_v3();
    drafter.architectures = vec!["DFlash2DraftModel".into()];
    let sub = drafter.dflash_config.as_mut().unwrap();
    sub.conv_kernel_size = 2;
    sub.conv_group_size = 16;
    sub.selector_rank = 256;
    sub.selector_top_k = 16;
    sub.selector_vocab_size = Some(248_077);
    drafter.layer_types = Some(
        QWEN38_FLASH_NEXT_DFLASH2_LAYER_TYPES
            .map(String::from)
            .to_vec(),
    );
    drafter.sliding_window = Some(4096);
    drafter
}

type TargetMutation = (&'static str, fn(&mut ModelConfig));
type DrafterMutation = (&'static str, fn(&mut DflashConfig));

const TARGET_MUTATIONS: &[TargetMutation] = &[
    ("identity", |_| {}),
    ("not qwen4", |t| t.model_type = "qwen3_next".into()),
    ("hidden 2048", |t| t.hidden_size = 2048),
    ("hc 1", |t| t.hc_count = 1),
    ("hc 2", |t| t.hc_count = 2),
    ("logical vocab", |t| t.vocab_size = 248_077),
    ("vocab 248076", |t| t.vocab_size = 248_076),
    ("vocab 248319", |t| t.vocab_size = 248_319),
    ("layers 47", |t| t.num_hidden_layers = 47),
];

fn sub(drafter: &mut DflashConfig) -> &mut crate::weight_loader::DflashSubConfig {
    drafter.dflash_config.as_mut().unwrap()
}

const DRAFTER_MUTATIONS: &[DrafterMutation] = &[
    ("identity", |_| {}),
    ("arch other", |d| d.architectures = vec!["Other".into()]),
    ("hidden", |d| d.hidden_size = 2048),
    ("inter", |d| d.intermediate_size += 1),
    ("layers", |d| d.num_hidden_layers = 5),
    ("target layers", |d| d.num_target_layers = 64),
    ("heads", |d| d.num_attention_heads = 32),
    ("kv", |d| d.num_key_value_heads = 8),
    ("head_dim", |d| d.head_dim = 256),
    ("vocab", |d| d.vocab_size = 248_077),
    ("model_type", |d| d.model_type = None),
    ("root implicit", |d| d.root_block_size_explicit = false),
    ("root block 32", |d| d.block_size = 32),
    ("nested block 32", |d| sub(d).block_size = Some(32)),
    ("causal", |d| d.is_causal = None),
    ("tie", |d| d.tie_word_embeddings = true),
    ("draft vocab", |d| d.draft_vocab_size = Some(248_320)),
    ("markov", |d| d.markov_rank = 256),
    ("mask", |d| sub(d).mask_token_id = 248_070),
    ("tap", |d| sub(d).target_layer_ids[1] = 8),
    ("projector", |d| sub(d).projector_type = Some("mlp".into())),
    ("fc_layernorm", |d| sub(d).fc_layernorm = true),
    ("conv", |d| sub(d).conv_kernel_size = 3),
    ("group", |d| sub(d).conv_group_size = 8),
    ("rank", |d| sub(d).selector_rank = 128),
    ("top_k", |d| sub(d).selector_top_k = 8),
    ("selector vocab physical", |d| {
        sub(d).selector_vocab_size = Some(248_320)
    }),
    ("selector vocab none", |d| sub(d).selector_vocab_size = None),
    ("swa", |d| d.sliding_window = Some(2048)),
    ("layer types", |d| d.layer_types = None),
];

#[test]
fn native_pairing_matches_the_legacy_check_over_target_and_drafter_mutations() {
    for (fixture, base) in [("v3", native_v3()), ("dflash2", native_dflash2())] {
        for (target_name, mutate_target) in TARGET_MUTATIONS {
            for (drafter_name, mutate_drafter) in DRAFTER_MUTATIONS {
                let mut target = flash_next_target();
                mutate_target(&mut target);
                let mut drafter = base.clone();
                mutate_drafter(&mut drafter);
                assert_eq!(
                    is_native_qwen38_flash_next_drafter(&target, &drafter),
                    legacy_is_native(&target, &drafter),
                    "{fixture} / target {target_name} / drafter {drafter_name}"
                );
            }
        }
    }
}

#[test]
fn native_pairing_resolves_the_legacy_capture_mode_width_and_vocab() {
    let target = flash_next_target();
    for drafter in [native_v3(), native_dflash2()] {
        let profile = native_qwen38_flash_next_profile(&target, &drafter).expect("native pair");
        assert_eq!(
            runtime_capture_mode(profile).unwrap(),
            DflashCaptureMode::Qwen4HyperProjected
        );
        // Unqualified: the ATLAS_EXPERIMENTAL_NATIVE_QWEN4_DFLASH gate stays.
        assert!(!profile.qualified);
        for gamma in 0..=40 {
            assert_eq!(
                validate_native_qwen4_dflash_width(&drafter, Some(gamma)).is_ok(),
                gamma == QWEN38_FLASH_NEXT_DRAFT_GAMMA,
                "gamma {gamma}"
            );
        }
        assert!(validate_native_qwen4_dflash_width(&drafter, None).is_ok());
    }
    let dflash2 = native_dflash2();
    for requested in [0u32, 1, 100_000, 248_076, 248_077, 248_078, 248_320] {
        let resolved = resolve_native_dflash2_proposal_vocab(&dflash2, requested);
        let legacy_ok = requested == 0 || requested as usize == QWEN38_FLASH_NEXT_SELECTOR_VOCAB;
        assert_eq!(resolved.is_ok(), legacy_ok, "requested {requested}");
        if legacy_ok {
            assert_eq!(resolved.unwrap(), QWEN38_FLASH_NEXT_SELECTOR_VOCAB);
        }
    }
    assert!(resolve_native_dflash2_proposal_vocab(&native_v3(), 0).is_err());
}

#[test]
fn fp32_residual_targets_cannot_pair_a_bf16_capture() {
    let mut gemma = ModelConfig::qwen3_next_80b_nvfp4();
    gemma.model_type = "gemma4".into();
    gemma.num_experts = 0;
    for target in [gemma, flash_next_target()] {
        // `use_fp32_residual` honours ATLAS_GEMMA4_FP32 / ATLAS_FP32_RESIDUAL,
        // so assert the relation rather than one environment's outcome.
        let result = validate_dflash_capture_residual(&target);
        assert_eq!(
            result.is_ok(),
            !target.use_fp32_residual(),
            "{}",
            target.model_type
        );
        if let Err(error) = result {
            assert!(error.to_string().contains("FP32"), "{error:#}");
        }
    }
}
