// SPDX-License-Identifier: AGPL-3.0-only

//! Source-contract tripwires for the Qwen4 terminal-normalization dispatch.
//! Greedy token identity cannot catch a logit-scale bug, so no model path may
//! invoke the stored model-level final norm outside the one audited helper.

use std::fs;
use std::path::{Path, PathBuf};

fn rust_files(root: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).expect("read model source directory") {
        let path = entry.expect("source entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn admission_is_first_statement(source: &str, function: &str, admission: &str) -> bool {
    let source = compact(source);
    let Some((_, signature)) = source.split_once(&format!("fn{function}(")) else {
        return false;
    };
    let Some((_, body)) = signature.split_once('{') else {
        return false;
    };
    body.starts_with(admission)
}

fn partial_head_entrypoints() -> [(&'static str, &'static str, &'static str); 3] {
    [
        (
            // 295a6aaa1 moved the body into run_mtp_propose_owned_inner; the outer
            // wrapper is pinned separately to host-only checks before delegating.
            "impl_b3.rs",
            "run_mtp_propose_owned_inner",
            "super::final_norm::check_partial_head_admission(&self.config,super::final_norm::PartialHead::EarlyExit,Self::early_exit_enabled(),)?;",
        ),
        (
            "early_exit.rs",
            "early_exit_propose",
            "super::final_norm::check_partial_head_admission(&self.config,super::final_norm::PartialHead::EarlyExit,true,)?;",
        ),
        (
            "trait_impl/verify_d.rs",
            "decode_verify_graphed_kgamma_dispatch",
            "super::super::final_norm::check_partial_head_admission(&self.config,super::super::final_norm::PartialHead::LayerResolutionProbe,layer_resolution_probe_enabled(),)?;",
        ),
    ]
}

#[test]
fn unsupported_partial_heads_are_rejected_before_any_entrypoint_effects() {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/model");
    for (relative, function, admission) in partial_head_entrypoints() {
        let source =
            fs::read_to_string(model.join(relative)).expect("read partial-head entrypoint");
        assert!(
            admission_is_first_statement(&source, function, admission),
            "{relative}:{function} must admit the partial head before replay, checkpoint, GPU work, or row snapshots"
        );
    }
    let wrapper = fs::read_to_string(model.join("impl_b3.rs")).expect("read impl_b3.rs");
    assert!(
        admission_is_first_statement(
            &wrapper,
            "run_mtp_propose_inner",
            "crate::model::verify_failure::check()?;self.ensure_verify_sequence_idle(seq)?;letresult=self.run_mtp_propose_owned_inner(",
        ),
        "impl_b3.rs:run_mtp_propose_inner must only run host checks before the admitted body"
    );
}

#[test]
fn admission_contract_rejects_late_removed_and_ignored_guards() {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/model");
    for (relative, function, admission) in partial_head_entrypoints() {
        let source = compact(&fs::read_to_string(model.join(relative)).expect("read entrypoint"));
        for effect in [
            "self.checkpoint_ssm_states(seq)?;",
            "self.k8a_late_replay_if_pending(seq)?;",
            "self.sync_proposer_rotary(seq)?;",
            "self.ssm_h_to_f16_dispatch(seq)?;",
            "self.gpu.copy_d2h(hidden,&mutsnapshot)?;",
        ] {
            let late = source.replacen(admission, &format!("{effect}{admission}"), 1);
            assert!(
                !admission_is_first_statement(&late, function, admission),
                "{relative}:{function} accepted a guard after {effect}"
            );
        }
        for replacement in [
            String::new(),
            format!("let_={};", admission.trim_end_matches("?;")),
        ] {
            let bypass = source.replacen(admission, &replacement, 1);
            assert!(
                !admission_is_first_statement(&bypass, function, admission),
                "{relative}:{function} accepted a missing or ignored guard"
            );
        }
    }
}

#[test]
fn every_model_final_norm_weight_use_is_owned_by_the_dispatch_helper() {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/model");
    let helper = model.join("final_norm.rs");
    let mut files = Vec::new();
    rust_files(&model, &mut files);

    let mut offenders = Vec::new();
    for path in files {
        if path == helper {
            continue;
        }
        let source = fs::read_to_string(&path).expect("read model source");
        if source.contains("&self.final_norm") {
            offenders.push(path.strip_prefix(&model).unwrap().display().to_string());
        }
    }
    assert!(
        offenders.is_empty(),
        "model paths bypass final_norm_rows: {offenders:?}"
    );

    let helper_source = compact(&fs::read_to_string(helper).expect("read final-norm helper"));
    assert!(helper_source.contains("ifself.final_norm_identity{"));
    assert!(helper_source.contains("input==output"));
    assert!(helper_source.contains("ops::rms_norm("));
    assert!(helper_source.contains("&self.final_norm"));
}

#[test]
fn qwen4_serving_paths_mix_before_the_identity_dispatch() {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/model");
    for relative in [
        "trait_impl/decode_a.rs",
        "trait_impl/decode_qwen4_ple_graph.rs",
        "trait_impl/prefill_a.rs",
        "trait_impl/prefill_b/finalize_last.rs",
        "trait_impl/verify_b.rs",
        "trait_impl/verify_d.rs",
    ] {
        let source = compact(
            &fs::read_to_string(model.join(relative)).unwrap_or_else(|_| panic!("read {relative}")),
        );
        assert!(
            source.contains("qwen4_final_hidden") || source.contains("prepare_batched"),
            "{relative} lost the Qwen4 terminal mixer"
        );
        assert!(
            source.contains("final_norm_rows"),
            "{relative} bypasses the final-norm dispatch helper"
        );
    }
}

#[test]
fn construction_checks_the_identity_invariant_before_owning_the_mixer() {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/model");
    let constructor = compact(
        &fs::read_to_string(model.join("impl_a1.rs")).expect("read TransformerModel constructor"),
    );
    let check = constructor
        .find("check_final_norm_identity(&config,qwen4_final_mixer.as_ref())?")
        .expect("constructor must validate the identity invariant");
    let build = constructor
        .find("Ok(Self{")
        .expect("constructor must build the model after validation");
    assert!(
        check < build,
        "identity invariant must be checked before build"
    );
}
