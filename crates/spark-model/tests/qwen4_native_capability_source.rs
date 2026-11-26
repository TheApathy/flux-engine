// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only source contract: every native Qwen4 preflight check is either a
//! typed capability (evaluated before routing, so the default plan serializes
//! the frame instead of failing its rows) or a named invariant that serial
//! decode would not survive either. Source-only in FIX-02 (not compiled).

const DISPATCH: &str = include_str!("../src/model/trait_impl/decode_qwen4_native_batch.rs");
const CAPABILITY: &str = include_str!("../src/model/trait_impl/decode_qwen4_capability.rs");
const ROUTE: &str = include_str!("../src/model/trait_impl/decode_qwen4_route.rs");
const SSM: &str = include_str!("../src/layers/qwen3_ssm/qwen4_native_multi_seq.rs");
const SSM_TRAIT_IMPL: &str = include_str!("../src/layers/qwen3_ssm/mod.rs");
const ATTN_TRAIT_IMPL: &str = include_str!("../src/layers/qwen3_attention/trait_impl.rs");
const ATTN: &str =
    include_str!("../src/layers/qwen3_attention/trait_impl/multi_seq/qwen4_native.rs");
const PLE: &str = include_str!("../src/layers/qwen4_ple/runtime/verify_batch.rs");
const LAYER_TRAIT: &str = include_str!("../src/layer/transformer_layer.rs");

/// Dispatch messages that are not capabilities, with why.
const INVARIANTS: &[&str] = &[
    // Scheduler/model invariants: a violation is a bug, not a configuration.
    "native Qwen4 token and sequence row counts differ",
    "native Qwen4 token ID is outside the embedding vocabulary",
    "native Qwen4 decode requires flat rows with complete state and no swap lineage",
    "native Qwen4 PLE row-stride overflow",
    // State-dependent halves of capability-gated layer checks (aliasing).
    "native Qwen4 layer {layer_index} failed read-only preflight",
    // KV exhaustion: serial decode allocates the same blocks.
    "native Qwen4 decode needs {blocks_needed} KV blocks but only {} are free",
    // Lane 03 QSA rows: probed at load (`qwen4_native_qsa_capability`); what
    // remains per frame is state (pending verify rows, side-cache groups).
    "native Qwen4 layer {layer_index} failed QSA preflight",
    // The plan's `Profiling` process reason routes these frames serial first.
    "profiled Qwen4 requests must use the generic profile-aware decoder",
];

fn messages(source: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = source;
    while let Some(start) = rest.find('"') {
        let tail = &rest[start + 1..];
        let Some(end) = tail.find('"') else { break };
        let literal = &tail[..end];
        if literal.starts_with("native Qwen4") || literal.starts_with("profiled Qwen4") {
            found.push(literal.to_string());
        }
        rest = &tail[end + 1..];
    }
    found
}

#[test]
fn every_dispatch_preflight_message_is_a_capability_or_a_named_invariant() {
    let dispatch = DISPATCH
        .split("pub(in crate::model) fn decode_qwen4_native_batch_dispatch(")
        .nth(1)
        .expect("native dispatch");
    let pre_effect = dispatch
        .split_once("ensure_blocks_through_decode(")
        .expect("pre-effect half")
        .0;
    let checks = messages(pre_effect);
    assert!(checks.len() >= 25, "{checks:?}");
    for message in checks {
        assert!(
            CAPABILITY.contains(&message) || INVARIANTS.contains(&message.as_str()),
            "unclassified native preflight check: {message}"
        );
    }
}

#[test]
fn the_route_consults_capability_before_returning_a_native_frame() {
    let route = ROUTE
        .split("pub(super) fn qwen4_frame_route(")
        .nth(1)
        .expect("frame route");
    let gate = route
        .find("self.qwen4_native_capability(seqs.len())")
        .expect("capability gate");
    let gap = route
        .find("native_gap_route(&plan, seqs.len(), &gap)")
        .expect("typed gap route");
    assert!(gate < gap);
    assert!(CAPABILITY.contains("get_or_init(|| self.qwen4_native_capability_uncached(rows))"));
}

#[test]
fn layer_capabilities_cover_the_state_independent_preflight_halves() {
    assert!(LAYER_TRAIT.contains("fn qwen4_native_capability("));
    assert!(LAYER_TRAIT.contains("SerialReason::UnsupportedLayer"));
    for layer in [ATTN_TRAIT_IMPL, SSM_TRAIT_IMPL] {
        let capability = layer
            .split("fn qwen4_native_capability(")
            .nth(1)
            .expect("layer capability");
        let moe = capability
            .find("preflight_qwen4_native_exact_rows(num_seqs, ctx)")
            .expect("MoE half");
        let own = capability
            .find("SerialReason::AttentionLayer")
            .or_else(|| capability.find("SerialReason::SsmLayer"))
            .expect("layer half");
        assert!(capability.contains("SerialReason::MoeExactRowsUnavailable"));
        assert!(moe < own);
    }
    assert!(ATTN_TRAIT_IMPL.contains("self.preflight_qwen4_native_target_core(num_seqs, ctx)"));
    // The attention preflight runs the same core after its state checks.
    assert!(ATTN.contains("self.preflight_qwen4_native_target_core(num_seqs, ctx)"));
    assert!(
        ATTN.contains(
            "pub(in crate::layers::qwen3_attention) fn preflight_qwen4_native_target_core("
        )
    );

    // SSM: the preflight keeps only state checks after the capability call.
    let preflight = SSM
        .split("pub(super) fn preflight_qwen4_native_multi_seq(")
        .nth(1)
        .expect("SSM preflight")
        .split("pub(super) fn qwen4_native_multi_seq_capability(")
        .next()
        .unwrap();
    let call = preflight
        .find("self.qwen4_native_multi_seq_capability(rows, ctx)?")
        .expect("capability call");
    let alias = preflight
        .find("aliases another request's recurrent state")
        .expect("alias invariant");
    assert!(call < alias);
    for moved in [
        "exact_projection_route(",
        "preflight_qwen4_native_exact_rows(rows, ctx)",
        "staging exceeds its arena",
    ] {
        assert!(
            !preflight.contains(moved),
            "{moved} stayed in the state preflight"
        );
    }
    let capability = SSM
        .split("pub(super) fn qwen4_native_multi_seq_capability(")
        .nth(1)
        .expect("SSM capability");
    for check in [
        "native_ssm_ptr_cache_selected()?",
        "exceeds pointer-table capacity",
        "sequential NVFP4 QKVZ",
        "recurrent kernel set is incomplete",
        "exact_projection_route(",
        "K1 projection fallback kernel",
        "self.ffn.preflight_qwen4_native_exact_rows(rows, ctx)?",
        "staging exceeds its arena",
    ] {
        assert!(capability.contains(check), "SSM capability lacks {check}");
    }

    // PLE: capability without the poison checks, which fail serial too.
    let ple = PLE
        .split("pub(crate) fn native_decode_capability(")
        .nth(1)
        .expect("PLE capability")
        .split("pub fn forward_active_decode_rows(")
        .next()
        .unwrap();
    assert!(ple.contains("validate_active_decode_layout("));
    assert!(ple.contains("kernel set is incomplete"));
    assert!(!ple.contains("poisoned"));
}

#[test]
fn capability_names_the_checks_restrictions_md_listed() {
    for check in [
        "self.verify_lmhead_vocab() as usize == config.vocab_size",
        "R::LmHeadVocabCap",
        "!self.use_fp32_logits",
        "R::Fp32Logits",
        "w4a16_exact_lm_head_kernels.route_for_rows(rows as u32)",
        "R::LmHeadRowTier",
        "final_mixer\n            .preflight_qwen4_native_batched(rows, false)",
        "R::FinalMixer",
        "config.ple_layer_ids.as_slice() == [2]",
        "layer\n                .qwen4_native_capability(rows, &ctx)",
    ] {
        assert!(CAPABILITY.contains(check), "capability lacks {check:?}");
    }
}
