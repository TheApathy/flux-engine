// SPDX-License-Identifier: AGPL-3.0-only

#[test]
fn qkv16_packs_before_exact_projection_and_shipping_attention() {
    let route = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_only.rs");
    let pack = route.find("attn_hyper.pack_saved_tile_or_rows(").unwrap();
    let project = route.find("self.qwen4_k5_project_qkv_exact(").unwrap();
    let shipping = route
        .find("self.attention_forward_preprojected_raw(")
        .unwrap();
    assert!(pack < project && project < shipping);
}

#[test]
fn qkv16_batches_exact_o_projection_after_ordered_attention() {
    let route = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_only.rs");
    let raw = route
        .find("self.attention_forward_preprojected_raw(")
        .unwrap();
    let retain = route[raw..]
        .find("raw_outputs.offset(local * attn_row_bytes)")
        .unwrap()
        + raw;
    let project = route[retain..]
        .find("ops::w4a16_gemv_batch_logits_exact_with(")
        .unwrap()
        + retain;
    let inject = route[project..]
        .find("attn_hyper.inject_saved_tile_or_rows(")
        .unwrap()
        + project;
    assert!(raw < retain && retain < project && project < inject);
}

#[test]
fn qwen4_batch_core_uses_scaled_yarn_before_ordinary_rope() {
    let core = include_str!("../src/layers/qwen3_attention/prefill/cache_skip.rs");
    let qwen4 = core.find("!self.qwen4_yarn_inv_freq.is_null()").unwrap();
    let scaled = core[qwen4..].find("ops::rope_yarn_scaled(").unwrap();
    let ordinary = core[qwen4..].find("ops::rope(").unwrap();
    assert!(
        scaled < ordinary,
        "Qwen4 scaled YaRN must precede ordinary RoPE"
    );
}
