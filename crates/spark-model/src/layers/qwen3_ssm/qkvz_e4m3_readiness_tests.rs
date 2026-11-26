// SPDX-License-Identifier: AGPL-3.0-only

use super::trait_prefill_helper::ssm_proj_act_scratch_capacity;

#[test]
fn eager_qkvz_cache_publication_is_fenced_and_fail_closed() {
    let helper = include_str!("trait_prefill_helper.rs");
    let start = helper
        .find("pub(crate) fn prepare_qkvz_e4m3_cache(")
        .expect("construction-time QKVZ cache preparation must exist");
    let end = helper[start..]
        .find("fn prepared_qkvz_weight_e4m3(")
        .map(|offset| start + offset)
        .expect("runtime prepared-cache accessor must exist");
    let prepare = &helper[start..end];

    let allocate = prepare.find(".alloc(bytes)").expect("cache allocation");
    let launch = prepare
        .find("ops::dequant_nvfp4_to_e4m3(")
        .expect("the established NVFP4-to-e4m3 conversion");
    let fence = prepare
        .find("gpu.synchronize(stream)")
        .expect("construction-time stream fence");
    let publish = prepare
        .find("*slot = Some(ptr)")
        .expect("cache publication");

    assert!(allocate < launch && launch < fence && fence < publish);
    assert!(prepare.contains("requires dequant_nvfp4_to_e4m3"));
    assert!(prepare.contains("if let Err(error) = ops::dequant_nvfp4_to_e4m3("));
    assert!(prepare.contains("if let Err(error) = gpu.synchronize(stream)"));
    assert_eq!(prepare.matches("gpu.free(ptr)").count(), 2);

    let runtime = &helper[end..];
    assert!(runtime.contains("cache was not eagerly prepared during model load"));
    assert!(!runtime.contains("materialised once on first"));
}

#[test]
fn dense_qwen_loader_prepares_every_ssm_cache_after_construction() {
    let loader = include_str!("../../weight_loader/qwen35_dense.rs");
    let constructor = loader
        .find("let mut layer = Qwen3SsmLayer::new_sequential_with_splitk(")
        .expect("dense Qwen SSM construction");
    let prepare = loader[constructor..]
        .find(".prepare_qkvz_e4m3_cache(gpu, config, stream)")
        .map(|offset| constructor + offset)
        .expect("eager QKVZ cache preparation");
    let publish_layer = loader[constructor..]
        .find("layers.push(Box::new(layer))")
        .map(|offset| constructor + offset)
        .expect("constructed layer publication");

    assert!(constructor < prepare && prepare < publish_layer);
    assert!(loader.contains("let cublaslt_ssm_proj_requested ="));
    assert!(loader[constructor..prepare].contains("if cublaslt_ssm_proj_requested"));
}

#[test]
fn eager_cache_change_keeps_e19_widest_projection_scratch_contract() {
    const ROWS: usize = 8_193;
    const HIDDEN: usize = 5_120;
    const VALUE_HEADS: usize = 48;
    const VALUE_HEAD_DIM: usize = 128;
    const VALUE_DIM: usize = VALUE_HEADS * VALUE_HEAD_DIM;

    assert_eq!(
        ssm_proj_act_scratch_capacity(ROWS, HIDDEN, VALUE_HEADS, VALUE_HEAD_DIM),
        ROWS.checked_mul(HIDDEN.max(VALUE_DIM))
    );
    let helper = include_str!("trait_prefill_helper.rs");
    assert!(helper.contains("arena_rows.checked_mul(hidden_size.max(value_dim))"));
}

#[test]
fn take_bf16_qkvz_self_protects_when_gdn_bf16_alias_is_live() {
    let module = include_str!("mod.rs");
    let start = module
        .find("pub(crate) fn take_bf16_qkvz(")
        .expect("take_bf16_qkvz must exist");
    let end = module[start..]
        .find("\n    }\n")
        .map(|offset| start + offset)
        .expect("take_bf16_qkvz body must end");
    let body = &module[start..end];
    assert!(
        body.contains("self.qkvz_dense_decode_only.is_some()"),
        "take_bf16_qkvz must refuse while the GDN_BF16 decode-only alias is live: {body}"
    );
}

#[test]
fn loader_skips_bf16_qkvz_release_when_gdn_bf16_is_engaged() {
    let loader = include_str!("../../weight_loader/qwen35/load_layers/linear_attn_arms.rs");
    assert!(
        loader.contains("if release_bf16_qkvz && !gdn_bf16_engaged {"),
        "ATLAS_QWEN4_RELEASE_BF16_QKVZ must become a no-op once ATLAS_QWEN4_GDN_BF16 is engaged"
    );
}
