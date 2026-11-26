// SPDX-License-Identifier: AGPL-3.0-only

//! Source-level contract for prefill failure isolation.
//!
//! These checks intentionally avoid CUDA. They make the scheduler's fatal
//! boundaries and the production `Model` stream-drain implementations visible
//! to ordinary CPU CI.

fn compact(source: &str) -> String {
    source.split_whitespace().collect()
}

#[test]
fn model_requires_stream_synchronization() {
    let model_trait = compact(include_str!("../../spark-model/src/traits/model.rs"));
    assert!(model_trait.contains("fnsynchronize_stream(&self,stream:u64)->Result<()>;"));

    for production_impl in [
        include_str!("../../spark-model/src/model/trait_impl/mod.rs"),
        include_str!("../../spark-model/src/model/dsv41.rs"),
        include_str!("../../spark-model/src/model/glm53/model_trait.rs"),
        include_str!("../../spark-model/src/model/glm53/model_trait_exl3.rs"),
    ] {
        assert!(production_impl.contains("fn synchronize_stream("));
    }
}

#[test]
fn scheduler_routes_prefill_failures_through_the_barrier() {
    let scheduler = include_str!("../src/scheduler/mod.rs");
    let first_chunk = include_str!("../src/scheduler/prefill_a_step.rs");
    let continuation = include_str!("../src/scheduler/phase_continue_prefills.rs");
    let standard = include_str!("../src/scheduler/phase_continue_prefills/run_standard.rs");
    let batch = include_str!("../src/scheduler/phase_continue_prefills/run_batched_prefill.rs");
    let mixed_batch = include_str!("../src/scheduler/phase_continue_prefills/run_batched_mixed.rs");

    assert!(scheduler.contains("mod prefill_stream_barrier;"));
    assert!(first_chunk.contains("drain_before_prefill_cleanup("));
    assert!(standard.contains("drain_before_prefill_cleanup("));
    assert!(batch.contains("drain_before_prefill_cleanup("));
    assert!(standard.contains("abort_after_mixed_forward_error("));
    assert!(mixed_batch.contains("abort_after_mixed_forward_error("));
    assert!(continuation.contains("drain_before_prefill_cleanup("));
    assert!(!continuation.contains("falling back to chunked"));
}

#[test]
fn qwen4_is_excluded_before_non_replayable_two_phase_prefill() {
    let model_trait = compact(include_str!("../../spark-model/src/traits/model.rs"));
    let transformer = compact(include_str!(
        "../../spark-model/src/model/trait_impl/mod.rs"
    ));
    let continuation = compact(include_str!("../src/scheduler/phase_continue_prefills.rs"));

    assert!(model_trait.contains("fnsupports_prefill_twophase(&self)->bool"));
    let capability = transformer
        .split_once("fnsupports_prefill_twophase(&self)->bool")
        .unwrap()
        .1
        .split_once("fnprefill_twophase(")
        .unwrap()
        .0;
    assert!(capability.contains("!self.config.is_qwen4_exp()"));
    assert!(continuation.contains("model.supports_prefill_twophase()"));
    assert!(continuation.contains("requestwillnotbereplayed"));
}

#[test]
fn successful_handoffs_do_not_ignore_event_errors() {
    for source in [
        include_str!("../src/scheduler/mod.rs"),
        include_str!("../src/scheduler/prefill_a_step.rs"),
        include_str!("../src/scheduler/phase_continue_prefills.rs"),
        include_str!("../src/scheduler/phase_continue_prefills/run_standard.rs"),
        include_str!("../src/scheduler/phase_continue_prefills/run_batched_prefill.rs"),
        include_str!("../src/scheduler/phase_continue_prefills/run_batched_mixed.rs"),
    ] {
        assert!(!source.contains("let _ = model.record_event(prefill_event"));
        assert!(!source.contains("let _ = model.stream_wait_event(model.default_stream()"));
    }
}
