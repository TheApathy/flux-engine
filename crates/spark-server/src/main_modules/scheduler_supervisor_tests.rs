// SPDX-License-Identifier: AGPL-3.0-only

use super::run_supervised;
use atlas_core::fault::FaultLatch;

#[test]
fn str_panic_publishes_its_reason() {
    let latch = FaultLatch::new();
    assert!(run_supervised(&latch, || panic!("borrowed payload")));
    let reason = latch.fault().expect("fault must be latched");
    assert!(reason.contains("scheduler thread panicked"));
    assert!(reason.contains("borrowed payload"));
}

#[test]
fn string_panic_publishes_its_reason() {
    let latch = FaultLatch::new();
    assert!(run_supervised(&latch, || {
        std::panic::panic_any(String::from("owned payload"));
    }));
    let reason = latch.fault().expect("fault must be latched");
    assert!(reason.contains("scheduler thread panicked"));
    assert!(reason.contains("owned payload"));
}

#[test]
fn clean_body_leaves_the_latch_clear() {
    let latch = FaultLatch::new();
    assert!(!run_supervised(&latch, || ()));
    assert!(!latch.is_faulted());
}

#[test]
fn scheduler_panic_cannot_overwrite_an_earlier_fault() {
    let latch = FaultLatch::new();
    assert!(latch.latch("original fault"));
    assert!(run_supervised(&latch, || panic!("later panic")));
    assert_eq!(latch.fault(), Some("original fault"));
}

#[test]
fn production_scheduler_and_router_are_both_wired_to_the_fault_latch() {
    let load = include_str!("serve_load.rs");
    let router = include_str!("serve_router.rs");
    let spawn = load
        .find("let scheduler_handle = std::thread::spawn")
        .expect("scheduler spawn");
    let supervise = load
        .find("scheduler_supervisor::run_supervised(atlas_core::fault::global()")
        .expect("panic supervisor");
    let run = load[supervise..]
        .find("scheduler::run(")
        .map(|offset| supervise + offset)
        .expect("scheduler body");

    assert!(spawn < supervise && supervise < run);
    assert!(router.contains(".layer(axum::middleware::from_fn(fault_middleware))"));
}
