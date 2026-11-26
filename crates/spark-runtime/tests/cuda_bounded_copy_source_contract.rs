// SPDX-License-Identifier: AGPL-3.0-only
//! Source-only checks for the opt-in bounded host-copy failure contract.

use std::fs;
use std::path::PathBuf;

fn source(relative: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(root.join(relative)).unwrap()
}

fn method_body<'a>(source: &'a str, name: &str, next_name: &str) -> &'a str {
    let marker = format!("    fn {name}(");
    let start = source.find(&marker).expect("method start");
    let next_marker = format!("\n    fn {next_name}(");
    let relative_end = source[start..].find(&next_marker).expect("next method");
    &source[start..start + relative_end]
}

#[test]
fn bounded_copies_are_exact_opt_in_and_poison_after_uncertain_completion() {
    let copy = source("src/cuda_backend/bounded_copy.rs");
    assert!(copy.contains("ATLAS_BOUNDED_HOST_COPY"));
    assert!(copy.contains("ENGAGED ATLAS_BOUNDED_HOST_COPY"));
    assert!(copy.contains("value == Some(\"1\")"));
    assert!(copy.contains("static POISONED: AtomicBool"));
    assert!(copy.contains("static COPY_GATE: Mutex<()>"));
    assert!(copy.contains("bounded host copy disabled after uncertain DMA completion"));
    assert!(copy.contains("fn poison_state(state: &AtomicBool)"));
}

#[test]
fn event_is_destroyed_only_after_completion_and_timeouts_leak_it() {
    let copy = source("src/cuda_backend/bounded_copy.rs");
    let drain = copy.split("fn drain_locked(").nth(1).unwrap();
    assert_eq!(drain.matches("cuEventDestroy_v2(event)").count(), 1);
    let complete = drain.find("PollDecision::Complete => {").unwrap();
    let destroy = drain.find("cuEventDestroy_v2(event)").unwrap();
    let poison = drain.find("PollDecision::Poison => {").unwrap();
    let timeout = poison
        + drain[poison..]
            .find("event {event:#x} leaked, context poisoned")
            .unwrap();
    assert!(complete < destroy && destroy < poison && poison < timeout);
}

#[test]
fn async_stage_is_not_reused_until_event_completion() {
    let copy = source("src/cuda_backend/bounded_copy.rs");
    let h2d = copy.split("fn h2d_after_drain_locked(").nth(1).unwrap();
    let write = h2d
        .find("copy_nonoverlapping(src.as_ptr().add(off), stage, len)")
        .unwrap();
    let enqueue = h2d.find("cuMemcpyHtoDAsync_v2").unwrap();
    let wait = h2d.find("drain_locked(stream, what)?").unwrap();
    let advance = h2d.find("off += len").unwrap();
    assert!(write < enqueue && enqueue < wait && wait < advance);
}

#[test]
fn pair_and_group_share_one_bounded_producer_drain() {
    let copy = source("src/cuda_backend/bounded_copy.rs");
    let pair = copy.split("pub(super) fn d2h_pair(").nth(1).unwrap();
    let pair = pair.split("pub(super) fn h2d(").next().unwrap();
    assert_eq!(pair.matches("producer drain").count(), 1);
    assert_eq!(pair.matches("d2h_after_drain_locked(").count(), 2);

    let group = copy.split("pub(super) fn h2d_group(").nth(1).unwrap();
    assert_eq!(group.matches("prior-work drain").count(), 1);
    assert_eq!(group.matches("h2d_after_drain_locked(").count(), 1);
}

#[test]
fn cuda_methods_keep_legacy_paths_behind_early_opt_in_returns() {
    let gpu = source("src/cuda_backend/gpu_impl.rs");
    for (name, next, bounded_call, legacy_call) in [
        (
            "copy_h2d",
            "copy_d2h",
            "bounded_copy::h2d(",
            "cuMemcpyHtoD_v2(",
        ),
        (
            "copy_d2h",
            "copy_d2h_on_stream",
            "bounded_copy::d2h(",
            "cuMemcpyDtoH_v2(",
        ),
        (
            "copy_d2h_on_stream",
            "copy_d2h_pair_on_stream",
            "bounded_copy::d2h(",
            "cuMemcpyDtoH_v2(",
        ),
        (
            "copy_d2h_pair_on_stream",
            "copy_h2d_group_on_stream",
            "bounded_copy::d2h_pair(",
            "cuMemcpyDtoH_v2(",
        ),
        (
            "copy_h2d_group_on_stream",
            "copy_d2d",
            "bounded_copy::h2d_group(",
            "cuMemcpyHtoD_v2(",
        ),
    ] {
        let body = method_body(&gpu, name, next);
        let bounded = body.find(bounded_call).expect("bounded call");
        let legacy = body.find(legacy_call).expect("legacy call");
        assert!(bounded < legacy, "{name} must select bounded mode first");
        assert!(body[..bounded].contains("bounded_copy::enabled()"));
        assert!(body[..bounded].contains("ensure_bounded_wait_outside_capture"));
        assert!(body[..legacy].contains("return super::bounded_copy::"));
    }
    assert!(method_body(&gpu, "copy_h2d", "copy_d2h").contains("drain_legacy_h2d_if_selected()"));
    assert!(
        method_body(&gpu, "copy_h2d_group_on_stream", "copy_d2d")
            .contains("drain_legacy_h2d_if_selected()")
    );
}

#[test]
fn explicit_stream_waits_and_blocking_memset_use_the_same_deadline() {
    let gpu = source("src/cuda_backend/gpu_impl.rs");
    assert!(gpu.contains("fn ensure_bounded_wait_outside_capture(&self, stream: u64, what: &str)"));
    assert!(
        gpu.contains("ATLAS_BOUNDED_HOST_COPY refuses a blocking wait during CUDA graph capture")
    );

    let synchronize = method_body(&gpu, "synchronize", "synchronize_device");
    assert!(synchronize.contains("ensure_bounded_wait_outside_capture(stream, \"synchronize\")"));
    assert!(
        synchronize.find("bounded_copy::drain(stream, \"synchronize\")")
            < synchronize.find("cuStreamSynchronize(stream)")
    );

    let memset = method_body(&gpu, "memset", "memset_async");
    assert!(
        memset.find("ensure_bounded_wait_outside_capture(self.default_stream, \"memset\")")
            < memset.find("cuMemsetD8Async")
    );
    assert!(
        memset.find("bounded_copy::drain(self.default_stream, \"memset\")")
            < memset.find("cuStreamSynchronize(self.default_stream)")
    );
}

#[test]
fn blocking_d2d_uses_bounded_completion_without_changing_the_legacy_path() {
    let gpu = source("src/cuda_backend/gpu_impl.rs");
    let d2d = method_body(&gpu, "copy_d2d", "launch");
    let bounded = d2d.find("bounded_copy::d2d(").expect("bounded D2D call");
    let legacy_copy = d2d
        .find("cuMemcpyDtoDAsync_v2(")
        .expect("legacy D2D enqueue");
    let legacy_wait = d2d
        .find("cuStreamSynchronize(self.default_stream)")
        .expect("legacy D2D completion wait");
    assert!(d2d[..bounded].contains("bounded_copy::enabled()"));
    assert!(
        d2d[..bounded].contains("ensure_bounded_wait_outside_capture"),
        "bounded D2D must reject graph capture before enqueue"
    );
    assert!(bounded < legacy_copy && legacy_copy < legacy_wait);
    assert!(d2d[..legacy_copy].contains("return super::bounded_copy::d2d("));

    let copy = source("src/cuda_backend/bounded_copy.rs");
    let bounded_d2d = copy.split("pub(super) fn d2d(").nth(1).unwrap();
    let bounded_d2d = bounded_d2d.split("pub(super) fn d2h(").next().unwrap();
    let health = bounded_d2d.find("ensure_healthy()?").unwrap();
    let enqueue = bounded_d2d.find("cuMemcpyDtoDAsync_v2").unwrap();
    let wait = bounded_d2d.find("drain_locked(stream").unwrap();
    assert!(health < enqueue && enqueue < wait);
    assert!(bounded_d2d.contains("poison()"));
}

#[test]
fn cpu_state_tests_cover_default_bypass_pending_timeout_and_query_error() {
    let copy = source("src/cuda_backend/bounded_copy.rs");
    assert!(copy.contains("assert!(!enabled_from_value(None))"));
    assert!(copy.contains("assert!(!enabled_from_value(Some(\"0\")))"));
    assert!(copy.contains("assert!(enabled_from_value(Some(\"1\")))"));
    assert!(copy.contains("poll_decision(600, false), PollDecision::Pending"));
    assert!(copy.contains("poll_decision(600, true), PollDecision::Poison"));
    assert!(copy.contains("poll_decision(719, false), PollDecision::Poison"));
    assert!(copy.contains("ensure_state_healthy(&state).is_err()"));
}
