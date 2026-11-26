// SPDX-License-Identifier: AGPL-3.0-only

//! What can be tested about a swap without a GPU.
//!
//! The load itself needs hardware, so these cover the decisions made BEFORE
//! anything is released — which is deliberate, because those are exactly the
//! ones that must never reach the teardown. A refusal that fires late costs a
//! live server its model; a refusal that fires here costs nothing.
//!
//! These CPU tests cover refusal, cancellation, and source-order invariants.
//! They do NOT qualify process exec, CUDA teardown, or live serving; those
//! remain integration gates on real hardware.

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{
    acquire_swap_guard, admit_in_process_initial_load, carry_process_flags, refuse_if_cancelled,
    refuse_if_shutting_down, wait_for_sole_owner,
};

// ── the drain window ────────────────────────────────────────────────────────

#[test]
fn a_holder_that_lets_go_is_waited_for_rather_than_refused() {
    // An in-flight request is a legitimate holder. Refusing the swap the
    // instant one exists would make swapping impossible under any load.
    let state = Arc::new(0u32);
    let borrowed = state.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(150));
        drop(borrowed);
    });
    assert_eq!(wait_for_sole_owner(&state, Duration::from_secs(5)), 0);
}

#[test]
fn a_holder_that_never_lets_go_is_reported_not_waited_on_forever() {
    // The deadlock this exists to prevent: a leaked Arc keeps request_tx open,
    // so joining the scheduler never returns. Bounded wait, then say how many
    // are stuck.
    let state = Arc::new(0u32);
    let _leaked = state.clone();
    let began = Instant::now();
    assert_eq!(wait_for_sole_owner(&state, Duration::from_millis(200)), 1);
    assert!(began.elapsed() < Duration::from_secs(2), "bounded");
}

#[test]
fn an_unshared_state_is_released_without_waiting() {
    let state = Arc::new(0u32);
    let began = Instant::now();
    assert_eq!(wait_for_sole_owner(&state, Duration::from_secs(30)), 0);
    assert!(
        began.elapsed() < Duration::from_millis(50),
        "no sleep at all"
    );
}

// ── refusals that must fire before anything is torn down ────────────────────

#[test]
fn a_shutting_down_process_refuses_to_start_a_load() {
    assert!(refuse_if_shutting_down(true).is_err());
    assert!(refuse_if_shutting_down(false).is_ok());
}

#[test]
fn only_a_completely_empty_host_may_load_without_restart_argv() {
    assert!(admit_in_process_initial_load(false, false, false, false).is_ok());
    for occupied in [
        (true, false, false, false),
        (false, true, false, false),
        (false, false, true, false),
        (false, false, false, true),
    ] {
        let error = admit_in_process_initial_load(occupied.0, occupied.1, occupied.2, occupied.3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("exact restart argv is required"));
        assert!(error.contains("running model is untouched"));
    }
}

#[test]
fn recipe_exec_precedes_every_model_release_and_no_raw_join_remains() {
    let source = include_str!("model_swap.rs");
    let exec = source
        .find("return Err(super::swap_env::exec(")
        .expect("pre-teardown exec");
    let release = source
        .find("let carried = release_state(host, DRAIN_GRACE)?;")
        .expect("empty-host release");
    assert!(exec < release, "exec must happen before teardown");
    assert!(!source.contains(".join()"), "model swap must not raw-join");
}

#[test]
fn cancelled_launch_does_not_wait_for_the_swap_guard() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let host = crate::main_modules::model_host::ModelHost::empty();
    let _held = host.swap_guard();
    let cancel = AtomicBool::new(true);
    let began = Instant::now();
    assert!(refuse_if_cancelled(Some(&cancel)).is_err());
    assert!(acquire_swap_guard(&host, Some(&cancel)).is_err());
    assert!(cancel.load(Ordering::Acquire));
    assert!(began.elapsed() < Duration::from_millis(100));
}

// ── process flags survive a recipe ──────────────────────────────────────────

fn args_for(model: &str) -> crate::cli::ServeArgs {
    use clap::Parser;
    crate::cli::ServeArgs::parse_from(["serve", model])
}

#[test]
fn the_socket_is_not_moved_by_a_recipe() {
    // The listener is bound for the process lifetime, so a recipe's port is
    // unserveable by construction — carrying the live one is what keeps the
    // new model reachable at the address the operator is actually using.
    let mut previous = args_for("org/old");
    previous.bind = "0.0.0.0".into();
    previous.port = 9001;

    let mut next = args_for("org/new");
    next.bind = "127.0.0.1".into();
    next.port = 8000;

    carry_process_flags(&mut next, &previous);
    assert_eq!(next.bind, "0.0.0.0");
    assert_eq!(next.port, 9001);
    // and the thing the recipe DOES get to choose is untouched
    assert_eq!(next.model.as_deref(), Some("org/new"));
}

#[test]
fn request_dumping_is_not_silently_switched_off_by_a_swap() {
    // No recipe sets --dump, so without carrying it the file stays where it is
    // and is simply never written to again — the worst way for a diagnostic to
    // fail, because it looks like the traffic stopped.
    let mut previous = args_for("org/old");
    previous.dump = Some("/tmp/atlas-dump.jsonl".into());

    let mut next = args_for("org/new");
    assert!(next.dump.is_none());
    carry_process_flags(&mut next, &previous);
    assert_eq!(next.dump.as_deref(), Some("/tmp/atlas-dump.jsonl"));
}

#[test]
fn the_single_flight_recheck_compares_the_carried_argv_not_the_raw_recipe() {
    // The re-check after the guard is `previous == next`, and it runs AFTER
    // carrying. Comparing the raw recipe instead would make the two unequal
    // whenever any process flag was set, so every request queued behind a swap
    // would redo the load the winner had just finished — the stampede the
    // check exists to prevent.
    let mut previous = args_for("org/same");
    previous.port = 9001;
    previous.dump = Some("/tmp/d.jsonl".into());

    let mut next = args_for("org/same");
    assert_ne!(
        previous, next,
        "raw recipe differs on the process flags alone"
    );

    carry_process_flags(&mut next, &previous);
    assert_eq!(previous, next, "once carried, this is a no-op swap");
}

#[test]
fn a_different_checkpoint_is_still_a_real_swap_after_carrying() {
    let previous = args_for("org/old");
    let mut next = args_for("org/new");
    carry_process_flags(&mut next, &previous);
    assert_ne!(previous, next);
}
