// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only source contract for the first complete native Qwen4 model loop.

const LOOP: &str = include_str!("../src/model/trait_impl/decode_qwen4_native_batch.rs");
const ROUTE: &str = include_str!("../src/model/trait_impl/decode_a2.rs");
const FRAME_ROUTE: &str = include_str!("../src/model/trait_impl/decode_qwen4_route.rs");
const LAYER_TRAIT: &str = include_str!("../src/layer/transformer_layer.rs");
const ATTN: &str =
    include_str!("../src/layers/qwen3_attention/trait_impl/multi_seq/qwen4_native.rs");
const ATTN_TAIL: &str = include_str!("../src/layers/qwen3_attention/trait_impl/multi_seq/attn.rs");
const SSM: &str = include_str!("../src/layers/qwen3_ssm/qwen4_native_multi_seq.rs");
const FFN: &str = include_str!("../src/layers/mod.rs");
const MOE: &str = include_str!("../src/layers/moe/forward_k5.rs");
const PLE: &str = include_str!("../src/layers/qwen4_ple/runtime/verify_batch.rs");
const MODEL_DROP: &str = include_str!("../src/model/drop.rs");
const MODEL_IMPL_A3: &str = include_str!("../src/model/impl_a3.rs");
const HYPER: &str = include_str!("../src/layers/qwen4_hyper.rs");
const MODEL_TRAIT_IMPL: &str = include_str!("../src/model/trait_impl/mod.rs");
const PREFILL_SCHEDULER: &str =
    include_str!("../../spark-server/src/scheduler/phase_continue_prefills.rs");
const STANDARD_PREFILL: &str =
    include_str!("../../spark-server/src/scheduler/phase_continue_prefills/run_standard.rs");
const SCHEDULER_HELPERS: &str = include_str!("../../spark-server/src/scheduler/mod_helpers.rs");

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

#[test]
fn selected_frame_is_one_active_row_model_traversal_not_a_c1_loop() {
    let source = compact(LOOP);
    let method = source
        .split("pub(incrate::model)fndecode_qwen4_native_batch_dispatch(")
        .nth(1)
        .expect("native model loop");
    assert!(method.contains("self.embed_qwen4_batch(tokens,hidden,stream)?"));
    assert!(method.contains("for(layer_index,layer)inself.layers.iter().enumerate()"));
    assert!(method.contains("decode_qwen4_native_target_attention("));
    assert!(method.contains("decode_qwen4_native_target_mlp("));
    assert!(method.contains("layer.decode_qwen4_native_target_ssm("));
    assert!(!method.contains("layer.decode_multi_seq("));
    assert!(!method.contains("self.decode("));
    assert!(!method.contains("decode_batch_serial_rows("));
    assert!(!method.contains("padded_batch_n("));
}

#[test]
fn native_route_precedes_fp16_conversion_and_every_serial_frame_is_typed() {
    let source = compact(ROUTE);
    let select = source.find("self.qwen4_frame_route(seqs)?").unwrap();
    let native_arm = source
        .find("matches!(qwen4_route,Some(crate::qwen4_batch_policy::FrameRoute::Native))")
        .unwrap();
    let convert = source.find("self.ssm_h_to_f16_dispatch(s)?").unwrap();
    let serial = source
        .find("self.decode_batch_serial_rows(tokens,seqs)")
        .unwrap();
    assert!(select < native_arm && native_arm < convert && convert < serial);
    assert!(source.contains("self.config.is_qwen4_exp()"));
    // The pinned receipt grammar is unchanged; the reason is a second line.
    let frame = source.find("QWEN4_SERIAL_BATCH_FRAMErows={n}").unwrap();
    let reason = source.find("QWEN4_SERIAL_BATCH_REASONrows={n}").unwrap();
    assert!(frame < reason && reason < serial);
}

#[test]
fn profile_requests_bypass_native_and_reach_the_profile_aware_serial_route() {
    let route = compact(ROUTE);
    let profile_gate = route
        .find("self.qwen4_frame_route(seqs)?")
        .expect("typed frame route");
    let native = route
        .find("returnself.decode_qwen4_native_batch_dispatch(tokens,seqs)")
        .expect("native dispatch");
    let serial = route
        .find("returnself.decode_batch_serial_rows(tokens,seqs)")
        .expect("Qwen4 serial fallback");
    assert!(profile_gate < native && native < serial);

    let frame_route = compact(FRAME_ROUTE);
    let plan = frame_route
        .split("pub(super)fnqwen4_batch_plan_dispatch(")
        .nth(1)
        .expect("process batch plan")
        .split("pub(super)fnqwen4_frame_route(")
        .next()
        .unwrap();
    assert!(plan.contains("!qwen4_native_batch_runtime_eligible(self.profile)"));
    let native_loop = compact(LOOP);
    let direct_guard = native_loop
        .split("pub(incrate::model)fndecode_qwen4_native_batch_dispatch(")
        .nth(1)
        .expect("native dispatch body")
        .find("qwen4_native_batch_runtime_eligible(self.profile)")
        .expect("direct native profile rejection");
    let layer_preflight = native_loop
        .split("pub(incrate::model)fndecode_qwen4_native_batch_dispatch(")
        .nth(1)
        .unwrap()
        .find("preflight_qwen4_native_target(rows,&states,&preflight_ctx)")
        .expect("model-wide preflight");
    assert!(direct_guard < layer_preflight);
}

#[test]
fn qsa_bound_and_near_2k_rows_fall_back_before_native_dispatch() {
    let loop_source = compact(LOOP);
    let eligibility = loop_source
        .split_once("pub(super)fnqwen4_native_batch_request_eligible(")
        .expect("read-only request eligibility guard")
        .1
        .split_once("implTransformerModel")
        .expect("eligibility guard body")
        .0;
    assert!(eligibility.contains("!seq.qwen4_qsa_required"));
    assert!(eligibility.contains("seq.seq_len.checked_add(1)"));
    assert!(eligibility.contains("next<QSA_BOUNDARY"));
    assert!(!eligibility.contains("seq.seq_len+="));
    assert!(!eligibility.contains("seq.tokens.push("));
    assert!(!eligibility.contains("ensure_blocks_through_decode("));

    let frame_route = compact(FRAME_ROUTE);
    let frame_route = frame_route
        .split("pub(super)fnqwen4_frame_route(")
        .nth(1)
        .expect("typed frame route");
    assert!(frame_route.contains("qwen4_native_batch_selected(&plan,seqs.len())?"));
    assert!(frame_route.contains("qsa_required:seq.qwen4_qsa_required"));
    assert!(frame_route.contains("qwen4_batch_policy::route_frame(&plan,&facts)?"));
    assert!(frame_route.contains("qwen4_native_batch_request_eligible(seqs)"));
    assert!(!frame_route.contains("ensure_blocks_through_decode("));
    assert!(!frame_route.contains("std::mem::take("));

    let route = compact(ROUTE);
    let guard = route
        .find("self.qwen4_frame_route(seqs)?")
        .expect("request eligibility route guard");
    let native = route
        .find("returnself.decode_qwen4_native_batch_dispatch(tokens,seqs)")
        .expect("native dispatch");
    let serial = route
        .find("returnself.decode_batch_serial_rows(tokens,seqs)")
        .expect("Qwen4 serial fallback");
    assert!(guard < native && native < serial);
}

#[test]
fn c1_route_still_delegates_to_the_original_single_sequence_decoder() {
    let source = compact(ROUTE);
    let native_gate = source.find("ifn>1&&").expect("native C2-C32 width guard");
    let c1_start = source.find("ifn==1{").expect("C1 route");
    let c1_end = source[c1_start..]
        .find("}")
        .map(|offset| c1_start + offset)
        .expect("C1 route end");
    let c1 = &source[c1_start..=c1_end];
    assert!(native_gate < c1_start);
    assert!(c1.contains("self.decode(tokens[0],seqs[0],stream)?"));
    assert!(c1.contains("returnOk(self.decode_logits_ptr())"));
    assert!(!c1.contains("decode_qwen4_native_batch_dispatch"));
    assert!(!c1.contains("decode_batch_serial_rows"));
}

#[test]
fn output_is_one_full_vocab_logit_row_per_active_request() {
    let source = compact(LOOP);
    let full_vocab = source
        .find("self.verify_lmhead_vocab()asusize==self.config.vocab_size")
        .unwrap();
    let extent = source.find("checked_mul(self.config.vocab_size)").unwrap();
    let head = source
        .find("self.lm_head_qwen4_rows(final_hidden,rowsasu32,stream)")
        .unwrap();
    let commit = source.find("seq.tokens.push(tokens[row])").unwrap();
    assert!(full_vocab < extent && extent < head && head < commit);
    assert!(source.contains("logits_bytes<=arenas.logits"));

    let impl_a3 = compact(MODEL_IMPL_A3);
    let qwen4_head = impl_a3
        .split("pub(super)fnlm_head_qwen4_rows(")
        .nth(1)
        .expect("qwen4 multi-seq lm_head helper")
        .split("fndecode_tc_parity_lm_head(")
        .next()
        .unwrap();
    assert!(
        qwen4_head
            .contains("ifnum_tokens<=32{returnself.lm_head_batched(hidden,num_tokens,stream);}")
    );
    assert!(
        qwen4_head
            .contains("(33..=crate::layers::QWEN4_NATIVE_MAX_ROWSasu32).contains(&num_tokens)")
    );
    assert!(qwen4_head.contains("ops::w4a16_gemv_batch_logits_exact("));
    assert!(qwen4_head.contains("ops::ExactLmHeadRoute::SerialK1(_)"));
}

#[test]
fn request_owned_ple_and_recurrent_state_never_use_padding_slots() {
    let source = compact(LOOP);
    let uniqueness = source
        .find("unique.insert(slot)")
        .expect("slot uniqueness gate");
    let take = source
        .find("std::mem::take(&mutseq.layer_states)")
        .expect("state ownership transfer");
    let ple_slot = source
        .find("slot_idx:seq.slot_idx")
        .expect("PLE request slot");
    let refs = source
        .find("extract_layer_refs(&mutall_layer_states,layer_index)")
        .expect("per-request recurrent refs");
    let restore = source
        .find("seq.layer_states=states")
        .expect("state restoration");
    assert!(uniqueness < take && take < ple_slot && ple_slot < refs && refs < restore);
    assert!(!source.contains("dummy_slot"));
    assert!(!source.contains("0..padded"));
}

#[test]
fn c2_c32_attention_is_followed_by_the_matching_native_mlp_half() {
    let layer_trait = compact(LAYER_TRAIT);
    let attn = compact(ATTN);
    assert!(layer_trait.contains("fndecode_qwen4_native_target_mlp("));
    assert!(attn.contains("(2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&num_seqs)"));
    assert!(attn.contains("mlp_hyper.prepare_batched("));
    assert!(attn.contains("forward_qwen4_native_target_rows(ffn_input,num_seqs"));
    assert!(!attn.contains("self.ffn.forward_k2(ffn_input"));
    assert!(!attn.contains("self.ffn.forward_k3(ffn_input"));
    assert!(attn.contains("mlp_hyper.inject_saved_batched("));
}

#[test]
fn c2_c32_admission_is_explicit_and_qsa_fails_before_effects() {
    let source = compact(LOOP);
    let validate = source.find("validate_active_rows(rows,").unwrap();
    let qsa = source
        .find("qsa_required.iter().any(|&required|required)")
        .unwrap();
    let exact_lm_head = source
        .find("w4a16_exact_lm_head_kernels.route_for_rows(rowsasu32)")
        .unwrap();
    let kv_lock = source.find("letmutkv_cache=self.kv_cache.lock()").unwrap();
    let embed = source
        .find("self.embed_qwen4_batch(tokens,hidden,stream)?")
        .unwrap();
    assert!(source.contains("rows>=2&&rows<=crate::layers::QWEN4_NATIVE_MAX_ROWS"));
    assert!(qsa < validate && validate < exact_lm_head);
    assert!(exact_lm_head < kv_lock && kv_lock < embed);
    assert!(source.contains("next>=QSA_BOUNDARY"));
    assert!(!source.contains("ATLAS_QWEN4_K5_NATIVE_MOE"));
}

#[test]
fn actual_layer_states_are_preflighted_model_wide_before_any_effect() {
    let source = compact(LOOP);
    let state_refs = source
        .find("seq.layer_states[layer_index].as_ref()")
        .expect("actual per-request state refs");
    let preflight = source
        .find("preflight_qwen4_native_target(rows,&states,&preflight_ctx)")
        .expect("whole-model layer preflight");
    let kv_lock = source
        .find("letmutkv_cache=self.kv_cache.lock()")
        .expect("KV lock");
    let allocate = source
        .find("ensure_blocks_through_decode(")
        .expect("KV allocation");
    let embed = source
        .find("self.embed_qwen4_batch(tokens,hidden,stream)?")
        .expect("embedding");
    let state_take = source
        .find("std::mem::take(&mutseq.layer_states)")
        .expect("request-state transfer");
    assert!(state_refs < preflight && preflight < kv_lock);
    assert!(kv_lock < allocate && preflight < embed && preflight < state_take);

    let trait_source = compact(LAYER_TRAIT);
    let hook = trait_source
        .split("fnpreflight_qwen4_native_target(")
        .nth(1)
        .expect("fail-closed layer hook")
        .split("///Qwen4target-onlynativecross-sequenceSSMstage.")
        .next()
        .unwrap();
    assert!(hook.contains("layerdoesnotsupportnativeQwen4targetpreflight"));
}

#[test]
fn ple_kv_and_capacity_admission_precede_every_native_effect() {
    let model = compact(LOOP);
    let dispatch = model
        .split("pub(incrate::model)fndecode_qwen4_native_batch_dispatch(")
        .nth(1)
        .expect("native dispatch body")
        .split("#[cfg(test)]")
        .next()
        .unwrap();
    let ple = dispatch
        .find("ple.preflight_active_decode_rows(")
        .expect("PLE read-only preflight");
    let embed_preflight = dispatch
        .find("self.preflight_embed_qwen4_batch(rows)?")
        .expect("batched embedding kernel preflight");
    let metadata_capacity = dispatch
        .find("metadata_bytes<=arenas.scratch")
        .expect("metadata scratch capacity preflight");
    let kv = dispatch
        .find("validate_kv_rows_before_allocation(")
        .expect("KV ownership preflight");
    let free = dispatch
        .find("blocks_needed<=kv_cache.num_free_blocks()")
        .expect("KV free-capacity preflight");
    let ownership = dispatch
        .find("validate_kv_block_ownership(")
        .expect("KV live exclusive-ownership preflight");
    let allocate = dispatch
        .find("ensure_blocks_through_decode(")
        .expect("first KV allocation");
    let embed = dispatch
        .find("self.embed_qwen4_batch(tokens,hidden,stream)?")
        .expect("embedding");
    let upload = dispatch
        .find("self.upload_batch_metadata_fixed(")
        .expect("metadata upload");
    let take = dispatch
        .find("std::mem::take(&mutseq.layer_states)")
        .expect("state ownership transfer");
    assert!(metadata_capacity < embed_preflight && embed_preflight < ple);
    assert!(ple < kv && kv < ownership && ownership < free);
    assert!(free < allocate);
    assert!(free < embed && free < upload && free < take);

    let attention = compact(ATTN);
    let reject_kgamma = attention
        .find("validate_native_paged_decode_selectors(crate::layers::flash_attn_kgamma_bf16_enabled())")
        .expect("BF16 K-gamma preflight");
    let qkv_capacity = attention
        .find("qkv_bytes<=arena.qkv_output")
        .expect("exact QKV capacity preflight");
    let gqa4 = attention
        .find("preflight_bf16_gqa4_kernel(")
        .expect("lazy GQA4 kernel preflight");
    let first_attention_effect = attention
        .find("prepare_batched_into(")
        .expect("first attention effect");
    assert!(reject_kgamma < first_attention_effect);
    assert!(qkv_capacity < first_attention_effect);
    assert!(gqa4 < first_attention_effect);

    let moe = compact(MOE);
    let moe_preflight = moe
        .split("pub(crate)fnpreflight_qwen4_native_exact_rows(")
        .nth(1)
        .expect("MoE preflight body")
        .split("///Truewhendecoderunstheexact")
        .next()
        .unwrap();
    assert!(moe_preflight.contains("qwen4_topk_warp::preflight("));
    assert!(moe_preflight.contains("routed_inter_bytes<=arena.expert_gate_out"));

    let ple_source = compact(PLE);
    let ple_preflight = ple_source
        .find("pub(crate)fnpreflight_active_decode_rows(")
        .expect("PLE preflight method");
    let ple_io = ple_source
        .find("reader.read_rows_windowed(")
        .expect("PLE I/O");
    assert!(ple_preflight < ple_io);

    let active_ple = ple_source
        .split("pubfnforward_active_decode_rows(")
        .nth(1)
        .unwrap()
        .split("///ApplyPLEto`tokens.len()`consecutiveverifyrows")
        .next()
        .unwrap();
    assert!(!active_ple.contains("Qwen4PleVerifyScratch::allocate("));
    let init = compact(MODEL_DROP);
    assert!(init.contains("ple.initialize_verify_scratch(gpu.as_ref())?"));

    let embedding = compact(MODEL_IMPL_A3);
    let embedding_preflight = embedding
        .split("pub(super)fnpreflight_embed_qwen4_batch(")
        .nth(1)
        .expect("embedding preflight body")
        .split("///Batchedtwinofcalling")
        .next()
        .unwrap();
    assert!(embedding_preflight.contains("qwen4_batched_embed_expand"));
    assert!(embedding_preflight.contains("batched_embed"));
    assert!(embedding_preflight.contains("qwen4_hc_expand_embedding"));
}

#[test]
fn c2_c32_moe_preflight_requires_the_c1_exact_unified_route() {
    let ffn = compact(FFN);
    assert!(ffn.contains("preflight_qwen4_native_exact_rows(rows,ctx)"));

    let moe = compact(MOE);
    let preflight = moe
        .split("pub(crate)fnpreflight_qwen4_native_exact_rows(")
        .nth(1)
        .expect("MoE exact-row preflight")
        .split("///Truewhendecoderunstheexact")
        .next()
        .unwrap();
    assert!(preflight.contains("self.t_lanes_rows_route()"));
    assert!(preflight.contains("(2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&rows)"));
    assert!(!preflight.contains("ATLAS_QWEN4_K5_NATIVE_MOE"));
    assert!(preflight.contains("moe_expert_gate_up_shared_t_lanes_rows_k"));
    assert!(preflight.contains("moe_expert_silu_down_shared_t_lanes_rows_k"));
    assert!(preflight.contains("moe_weighted_sum_blend_batch3"));
    assert!(preflight.contains("gate_ptrs_t"));
    assert!(preflight.contains("up_ptrs_t"));
    assert!(preflight.contains("down_ptrs_t"));

    let native_execution = ffn
        .split("pub(crate)fnforward_qwen4_native_target_rows(")
        .nth(1)
        .expect("dedicated native-target FFN entry")
        .split("///Compose")
        .next()
        .unwrap();
    assert!(native_execution.contains("self.preflight_qwen4_native_exact_rows(rows,ctx)?"));
    assert!(
        native_execution.contains("moe.forward_qwen4_native_target_rows(input,rows,ctx,stream)")
    );
    assert!(!native_execution.contains("forward_k2("));
    assert!(!native_execution.contains("forward_k3("));

    let legacy_verify = ffn
        .split("pubfnforward_qwen4_exact_rows(")
        .nth(1)
        .expect("legacy verifier exact-row entry");
    assert!(legacy_verify.contains("ATLAS_QWEN4_K5_NATIVE_MOE"));

    let attention = compact(ATTN);
    assert!(
        attention
            .find("preflight_qwen4_native_exact_rows(")
            .unwrap()
            < attention.find("prepare_batched_into(").unwrap()
    );

    let ssm = compact(SSM);
    let execution = ssm
        .split("pub(super)fndecode_qwen4_native_multi_seq")
        .nth(1)
        .unwrap()
        .split("pub(super)fnpreflight_qwen4_native_multi_seq")
        .next()
        .unwrap();
    assert!(
        execution
            .find("self.preflight_qwen4_native_multi_seq(")
            .unwrap()
            < execution.find("attn_hyper.prepare_batched(").unwrap()
    );
    let ssm_preflight = ssm
        .split("pub(super)fnpreflight_qwen4_native_multi_seq")
        .nth(1)
        .unwrap();
    assert!(ssm_preflight.contains("preflight_qwen4_native_exact_rows(rows,ctx)"));
    assert!(ssm.contains("forward_qwen4_native_target_rows(ffn_input,rows,ctx,stream)"));
}

#[test]
fn c10_c32_moe_tiles_active_rows_without_padding_or_c1_replay() {
    let moe = compact(MOE);
    let execution = moe
        .split("pub(crate)fnforward_qwen4_native_target_rows(")
        .nth(1)
        .expect("native target MoE execution")
        .split("///Truewhendecoderunstheexact")
        .next()
        .unwrap();
    assert!(execution.contains("whilerow<rows"));
    assert!(execution.contains("native_target_moe_tile_width(rows-row)"));
    assert!(execution.contains("input.offset(row*row_bytes)"));
    assert!(execution.contains("output.offset(row*row_bytes)"));
    assert!(execution.contains("self.forward_qwen4_exact_into("));
    assert!(!execution.contains("self.forward_qwen4_exact(input,1"));
    assert!(!execution.contains("dummy"));
    assert!(!execution.contains("padding"));
}

#[test]
fn projection_preflight_covers_exact_tiers_and_k1_fallbacks_before_effects() {
    let loop_source = compact(LOOP);
    let lm_route = loop_source
        .find("w4a16_exact_lm_head_kernels.route_for_rows(rowsasu32)")
        .unwrap();
    let model_effect = loop_source
        .find("letmutkv_cache=self.kv_cache.lock()")
        .unwrap();
    assert!(lm_route < model_effect);
    assert!(loop_source.contains("ExactLmHeadRoute::SerialK1(_)"));
    assert!(loop_source.contains("unsetATLAS_LM_HEAD_TC"));

    let attention = compact(ATTN);
    let o_route = attention
        .find("w4a16_exact_o_proj_kernels.route_for_rows(num_seqsasu32)")
        .unwrap();
    assert!(o_route < attention.find("prepare_batched_into(").unwrap());
    assert!(attention.contains("K1O-projectionfallbackkernel"));
    assert!(attention.contains("letserial_o_proj=crate::layers::attn_out_serial_enabled()"));
    assert!(attention.contains("ifserial_o_proj||matches!(o_route"));
    assert!(attention.contains("preflight_qwen4_native_tail_selectors(num_seqs)?"));

    let tail = compact(ATTN_TAIL);
    let selector_preflight = tail
        .split("pub(super)fnpreflight_qwen4_native_tail_selectors(")
        .nth(1)
        .unwrap()
        .split("fnms_o_proj_nvfp4_k1(")
        .next()
        .unwrap();
    assert!(selector_preflight.contains("attn_gate_batched_requested()?"));
    assert!(selector_preflight.contains("(5..=17).contains(&rows)"));
    assert!(selector_preflight.contains("sigmoid_gate_mul_batched_k"));
    assert!(selector_preflight.contains("attn_o_proj_m17_astage_requested()?"));

    let ssm = compact(SSM);
    let execution = ssm
        .split("pub(super)fndecode_qwen4_native_multi_seq")
        .nth(1)
        .unwrap()
        .split("pub(super)fnpreflight_qwen4_native_multi_seq")
        .next()
        .unwrap();
    assert!(
        execution
            .find("self.preflight_qwen4_native_multi_seq(")
            .unwrap()
            < execution.find("attn_hyper.prepare_batched(").unwrap()
    );
    let ssm_preflight = ssm
        .split("pub(super)fnpreflight_qwen4_native_multi_seq")
        .nth(1)
        .unwrap();
    assert!(ssm_preflight.contains("exact_projection_route("));
    assert!(ssm_preflight.contains("K1projectionfallbackkernel"));

    let hyper = compact(HYPER);
    let hyper_preflight = hyper
        .split("pub(crate)fnpreflight_qwen4_native_batched(")
        .nth(1)
        .unwrap()
        .split("#[allow(clippy::too_many_arguments)]")
        .next()
        .unwrap();
    assert!(hyper_preflight.contains("(2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&rows)"));
    assert!(hyper_preflight.contains("route_for_rows(rowsasu32)"));
    assert!(hyper_preflight.contains("ExactLmHeadRoute::Exact(_)"));
    assert!(hyper_preflight.contains("inject_saved_k"));
    assert!(attention.contains("attn_hyper.preflight_qwen4_native_batched(num_seqs,true)?"));
    assert!(attention.contains("mlp_hyper.preflight_qwen4_native_batched(num_seqs,true)?"));
    assert!(ssm_preflight.contains("attn_hyper.preflight_qwen4_native_batched(rows,true)?"));
    assert!(ssm_preflight.contains("mlp_hyper.preflight_qwen4_native_batched(rows,true)?"));
    assert!(loop_source.contains("final_mixer.preflight_qwen4_native_batched(rows,false)?"));
}

#[test]
fn qwen4_highway_capability_blocks_unsupported_scheduler_paths() {
    let model = compact(MODEL_TRAIT_IMPL);
    let hc_method = model
        .split_once("fnhc_mult(&self)->usize{")
        .expect("TransformerModel highway capability")
        .1
        .split_once("fnis_mla(")
        .expect("highway capability body")
        .0;
    assert!(hc_method.contains("ifself.config.is_qwen4_exp(){"));
    assert!(hc_method.contains("self.config.hc_count"));
    assert!(hc_method.contains("self.config.hc_mult"));

    let prefill = compact(PREFILL_SCHEDULER);
    assert!(prefill.contains("letserialize_hc=model.hc_mult()>0;"));
    assert!(prefill.contains("!serialize_hc&&prefilling.len()>=2"));

    let standard = compact(STANDARD_PREFILL);
    assert!(standard.contains("&&model.hc_mult()==0;"));

    let helpers = compact(SCHEDULER_HELPERS);
    assert!(helpers.contains("ifmodel.hc_mult()==0&&i<active.len()"));
}
