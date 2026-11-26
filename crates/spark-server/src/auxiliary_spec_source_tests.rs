// SPDX-License-Identifier: AGPL-3.0-only

const SPEC_STEP: &str = include_str!("scheduler/spec_step.rs");
const SCHEDULER: &str = include_str!("scheduler/mod.rs");

fn between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing start marker: {start}"))
        .1
        .split_once(end)
        .unwrap_or_else(|| panic!("missing end marker: {end}"))
        .0
}

#[test]
fn self_spec_uses_canonical_verify_and_async_commit_receipt() {
    let body = between(
        SPEC_STEP,
        "pub fn step_self_spec(",
        "pub fn abandon_self_spec_frame(",
    );
    let checkpoint = body
        .find("model.start_checkpoint_async(&mut a.seq)")
        .expect("self-spec must publish a canonical checkpoint");
    let checkpoint_sync = body[checkpoint..]
        .find("model.sync_secondary()")
        .map(|offset| checkpoint + offset)
        .expect("checkpoint must have a receipt wait");
    let verify = body
        .find("model.decode_verify_graphed_kgamma(")
        .expect("self-spec must use the per-token-intermediate verifier");
    let commit = body
        .find("model.commit_verify_state_async(")
        .expect("self-spec must commit the accepted recurrent prefix");
    let receipt = body
        .find("*sync_pending = true;")
        .expect("self-spec must publish its async commit receipt");

    assert!(checkpoint < checkpoint_sync);
    assert!(checkpoint_sync < verify);
    assert!(verify < commit);
    assert!(commit < receipt);
    assert!(!body.contains("model.rollback_ssm_states("));
    assert!(!body.contains("model.decode_verify(&"));
}

#[test]
fn scheduler_drains_self_spec_receipt_before_serial_fallback_and_retirement() {
    let loop_body = SCHEDULER
        .split_once("} else if use_self_speculative {")
        .expect("self-spec scheduler branch must exist")
        .1;
    let fallback = loop_body
        .find("else if abandon_self_spec_frame(")
        .expect("serial fallback must drain the self-spec receipt");
    let serial = loop_body[fallback..]
        .find("step_decode_only(")
        .map(|offset| fallback + offset)
        .expect("ineligible self-spec frame must fall back to serial decode");
    let finished = loop_body
        .find("if active.iter().any(|a| a.finished)")
        .expect("finished self-spec requests must have a retirement guard");
    let retirement_drain = loop_body[finished..]
        .find("abandon_self_spec_frame(")
        .map(|offset| finished + offset)
        .expect("retirement guard must drain the self-spec receipt");

    assert!(fallback < serial);
    assert!(serial < finished);
    assert!(finished < retirement_drain);
    assert!(SCHEDULER.contains("let mut self_spec_sync_pending = false;"));
}

#[test]
fn qwen4_auxiliary_modes_remain_fail_closed() {
    let load = include_str!("main_modules/serve_load.rs");
    let predicate = between(
        load,
        "fn qwen4_aux_speculation_unsupported(",
        "#[cfg(test)]\nmod qualification_tests",
    );
    assert!(predicate.contains("is_qwen4 && (self_speculative || ngram_speculative)"));
    assert!(load.contains("if qwen4_aux_speculation_unsupported("));
    assert!(load.contains(
        "--self-speculative/--ngram-speculative are not losslessly qualified for qwen4_exp"
    ));
    assert!(load.contains("args.ngram_speculative && !model.supports_ngram_speculative()"));
    let transformer = include_str!("../../spark-model/src/model/trait_impl/mod.rs");
    let capability = between(
        transformer,
        "fn supports_ngram_speculative(&self) -> bool {",
        "fn decode_draft(",
    );
    assert!(capability.contains("!self.config.is_qwen4_exp()"));
}
