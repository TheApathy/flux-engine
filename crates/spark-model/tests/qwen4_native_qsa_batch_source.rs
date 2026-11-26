// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only source contract for per-row QSA rows in the native Qwen4 frame
//! (`ATLAS_QWEN4_NATIVE_BATCH_QSA=1`). These checks establish no runtime
//! correctness; they pin the ownership and ordering the eager design relies on.

const LOOP: &str = include_str!("../src/model/trait_impl/decode_qwen4_native_batch.rs");
const SEAM: &str =
    include_str!("../src/layers/qwen3_attention/trait_impl/multi_seq/qwen4_native.rs");
const ROWS: &str = include_str!("../src/layers/qwen3_attention/decode/qwen4_native_qsa.rs");
const POLICY: &str = include_str!("../src/layers/qwen4_qsa/native_batch.rs");
const LAYER_TRAIT: &str = include_str!("../src/layer/transformer_layer.rs");

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn body<'a>(source: &'a str, start: &str) -> &'a str {
    source
        .split(start)
        .nth(1)
        .unwrap_or_else(|| panic!("missing {start}"))
}

#[test]
fn selector_off_and_qsa_free_frames_keep_the_dense_lane() {
    let model = compact(LOOP);
    assert!(model.contains(
        "ifmode==NativeQsaMode::Off||!qsa_required.iter().any(|&required|required){returnOk(None);}"
    ));
    assert!(
        model.contains(
            "ifletSome(routes)=&qsa_routes{layer.decode_qwen4_native_target_attention_qsa("
        )
    );
    assert!(model.contains("}else{layer.decode_qwen4_native_target_attention("));

    let seam = compact(SEAM);
    let inner = body(&seam, "fndecode_qwen4_native_target_attention_inner(");
    assert!(inner.contains("block_tables,None,ctx,stream,)"));
    assert!(seam.contains("_=>self.ms_phase_paged_decode(&c,kv_cache,metadata)?,"));
    assert!(seam.contains("None=>(seq_lens,metadata.qwen4_qsa_required),"));
}

#[test]
fn each_qsa_row_runs_the_c1_index_before_its_own_attention_core() {
    let rows = compact(ROWS);
    let method = body(&rows, "fnqwen4_native_qsa_rows(");
    let row_loop = method
        .find("for(row,(&route,&seq_len))inroutes.iter().zip(seq_lens).enumerate()")
        .expect("strict row-order loop");
    let view = method
        .find("row_metadata_view(meta,row,route.maintains_index())?")
        .expect("one-row metadata view");
    let index = method
        .find("self.qwen4_qsa_indices(")
        .expect("C1 QSA index");
    let sparse = method
        .find("qsa.sparse_attention_bf16(")
        .expect("C1 sparse core");
    let dense = method
        .find("self.run_paged_decode(")
        .expect("C1 dense core");
    assert!(row_loop < view && view < index && index < sparse && index < dense);
    assert!(method.contains("attn_metadata:Some(row_meta),..*ctx"));
    assert!(
        method.contains("row_meta.block_table,row_meta.seq_len,row_meta.max_blocks_per_seq,1,")
    );
    for batched in [
        "sparse_attention_bf16_rows(",
        "update_and_select_verify_rows(",
        "select_expand_rows",
        "overlap(",
    ] {
        assert!(!rows.contains(batched), "phase 0 must not use {batched}");
    }
    let policy = compact(POLICY);
    let view_fn = body(&policy, "fnrow_metadata_view(");
    assert!(view_fn.contains("num_seqs:1,..meta"));
}

#[test]
fn qsa_preflight_precedes_every_native_effect() {
    let model = compact(LOOP);
    let dispatch = body(
        &model,
        "pub(incrate::model)fndecode_qwen4_native_batch_dispatch(",
    );
    let free = dispatch
        .find("blocks_needed<=kv_cache.num_free_blocks()")
        .expect("KV free-capacity preflight");
    let qsa = dispatch
        .find("preflight_qwen4_native_target_qsa(routes,&seq_lens,&kv_cache)")
        .expect("model-wide QSA preflight");
    let allocate = dispatch
        .find("ensure_blocks_through_decode(")
        .expect("first KV allocation");
    assert!(free < qsa && qsa < allocate);

    let seam = compact(SEAM);
    let routed = body(&seam, "fndecode_qwen4_native_target_attention_routed(");
    let layer_qsa = routed
        .find("self.preflight_qwen4_native_qsa_rows(routes,seq_lens,kv_cache)?")
        .expect("layer QSA preflight");
    let first_effect = routed.find("prepare_batched_into(").expect("first effect");
    assert!(layer_qsa < first_effect);

    let policy = compact(POLICY);
    let preflight = body(&policy, "fnpreflight_native_rows(");
    for effect in ["gpu.alloc(", ".launch(", "self.cache(", "memset"] {
        assert!(
            !preflight.contains(effect),
            "QSA preflight performs {effect}"
        );
    }
}

#[test]
fn qsa_layer_hooks_fail_closed_by_default() {
    let layer_trait = compact(LAYER_TRAIT);
    assert!(layer_trait.contains("layerdoesnotsupportnativeQwen4QSArows"));
    assert!(layer_trait.contains("layerdoesnotsupportnativeQwen4QSAattention"));
}

#[test]
fn qsa_selector_is_validated_on_the_fallible_route_path() {
    let model = compact(LOOP);
    let selected = body(&model, "pub(super)fnqwen4_native_batch_selected(");
    let qsa = selected
        .find("resolve_native_batch_qsa_mode(selected,qsa_mode)")
        .expect("QSA selector validation");
    let width = selected
        .find("select_native_target_width(selected,rows)")
        .expect("width selection");
    assert!(qsa < width);
    let policy = compact(POLICY);
    assert!(policy.contains("Some(_)=>Err(\"ATLAS_QWEN4_NATIVE_BATCH_QSAmustbeexactvalue0or1\")"));
}

#[test]
fn receipts_label_qsa_frames_truthfully() {
    let model = compact(LOOP);
    assert!(model.contains("qsa=trueroutes={routes}graph=false"));
    assert!(model.contains("active_only=truetarget_only=trueqsa=falsegraph=false"));
    assert!(model.contains("QWEN4_NATIVE_QSA_ENGAGED"));
}
