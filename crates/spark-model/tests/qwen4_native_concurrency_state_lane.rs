// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only source contract for the native Qwen4 C2-C32 state lane.

const SSM: &str = include_str!("../src/layers/qwen3_ssm/qwen4_native_multi_seq.rs");
const SSM_DISPATCH: &str = include_str!("../src/layers/qwen3_ssm/trait_decode_multi_seq.rs");
const SSM_TRAIT_IMPL: &str = include_str!("../src/layers/qwen3_ssm/mod.rs");
const LAYER_TRAIT: &str = include_str!("../src/layer/transformer_layer.rs");
const NATIVE_LOOP: &str = include_str!("../src/model/trait_impl/decode_qwen4_native_batch.rs");
const PLE: &str = include_str!("../src/layers/qwen4_ple/runtime/verify_batch.rs");

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

#[test]
fn ssm_lane_uses_independent_state_pointer_tables_between_hyper_boundaries() {
    let source = compact(SSM);
    let preflight = source
        .find("self.preflight_qwen4_native_multi_seq(")
        .unwrap();
    let prepare = source.find("attn_hyper.prepare_batched(").unwrap();
    let state_core = source
        .find("self.ssm_forward_qwen4_native_multi_seq(")
        .unwrap();
    let pointer_table = source.find("self.refresh_multi_seq_ptr_table(").unwrap();
    let conv = source
        .find("ops::conv1d_update_l2norm_f32_multi_seq(")
        .unwrap();
    let gdn = source.find("ops::gdn_decode_f32_multi_seq(").unwrap();
    let inject = source.find("attn_hyper.inject_saved_batched(").unwrap();
    assert!(preflight < prepare && prepare < state_core);
    assert!(state_core < pointer_table && pointer_table < conv && conv < gdn);
    assert!(state_core < inject);
    assert!(source.contains("rows>=2&&rows<=crate::layers::QWEN4_NATIVE_MAX_ROWS"));
    assert!(source.contains("!ctx.graph_capture"));
    assert!(source.contains("rows<=self.ssm_multi_seq_ptr_max"));
    assert!(source.contains("!h_states.contains(&state.h_state.0)"));
    assert!(source.contains("!conv_states.contains(&state.conv_state.0)"));
}

#[test]
fn generic_multi_seq_dispatch_never_selects_the_native_qwen4_lane() {
    let dispatch = compact(SSM_DISPATCH);
    let method = dispatch
        .split("pub(super)fndecode_multi_seq_inner")
        .nth(1)
        .expect("generic SSM multi-sequence method");
    assert!(method.contains("letuse_batched="));
    assert!(method.contains("self.decode_multi_seq_per_seq_fallback("));
    assert!(!method.contains("decode_qwen4_native_multi_seq("));
    assert!(!method.contains("qwen4_attn_hyper"));
    assert!(!method.contains("qwen4_mlp_hyper"));
}

#[test]
fn native_ssm_is_reachable_only_through_a_dedicated_fail_closed_hook() {
    let layer_trait = compact(LAYER_TRAIT);
    let default_hook = layer_trait
        .split_once("fndecode_qwen4_native_target_ssm")
        .expect("native Qwen4 SSM trait hook")
        .1
        .split_once("///Qwen4target-onlynativecross-sequenceATTENTIONstage.")
        .expect("native Qwen4 SSM trait hook body")
        .0;
    assert!(default_hook.contains("layerdoesnotsupportnativeQwen4targetSSM"));

    let ssm_impl = compact(SSM_TRAIT_IMPL);
    let implementation = ssm_impl
        .split_once("fndecode_qwen4_native_target_ssm")
        .expect("Qwen3 SSM native hook implementation")
        .1
        .split_once("fndecode_batched(")
        .expect("Qwen3 SSM native hook body")
        .0;
    assert!(implementation.contains("self.decode_qwen4_native_multi_seq("));

    let native_loop = compact(NATIVE_LOOP);
    assert!(native_loop.contains("layer.decode_qwen4_native_target_ssm("));
    assert!(!native_loop.contains("layer.decode_multi_seq("));
}

#[test]
fn ple_lane_is_active_only_slot_indexed_and_never_walks_a_padded_width() {
    let source = compact(PLE);
    let method = source
        .split("pubfnforward_active_decode_rows(")
        .nth(1)
        .expect("active PLE seam");
    let before_verify = method
        .split("///ApplyPLEto`tokens.len()`consecutiveverifyrows")
        .next()
        .unwrap();
    assert!(before_verify.contains("rows:&[PleActiveDecodeRow<'_>]"));
    assert!(source.contains("(2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&rows)"));
    assert!(before_verify.contains("for(row_idx,row)inrows.iter().enumerate()"));
    assert!(before_verify.contains("row.slot_idx"));
    assert!(before_verify.contains("row.reset_state"));
    assert!(!before_verify.contains("padded_n"));
    assert!(!before_verify.contains("0..padded"));
    assert!(!before_verify.contains("dummy_slot"));
}

#[test]
fn ple_layout_rejects_aliasing_slots_before_any_gpu_submission() {
    let source = compact(PLE);
    let validator = source.find("fnvalidate_active_decode_layout(").unwrap();
    let method = source.find("pubfnforward_active_decode_rows(").unwrap();
    let first_effect = source[method..]
        .find("let_submit=self.scratch_submit.lock()")
        .unwrap()
        + method;
    let validation_call = source[method..]
        .find("self.preflight_active_decode_rows(")
        .unwrap()
        + method;
    assert!(validator < method);
    assert!(validation_call < first_effect);
    assert!(source.contains("!slots[..row].contains(&slot)"));
}
