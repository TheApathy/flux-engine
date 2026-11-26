// SPDX-License-Identifier: AGPL-3.0-only

//! Model lifecycle transitions with fail-closed restart boundaries.
//!
//! A scheduler owns the model and may have CUDA work or PLE I/O in flight.
//! There is no portable way to kill a Rust thread, so a live recipe switch must
//! not clear the host and then wait forever for that thread. The production TUI
//! path instead validates the complete prospective launch while the old model
//! is healthy, then `exec`s the exact recipe argv. POSIX replaces all old
//! threads and the CUDA context atomically; an exec error returns before any
//! state is released. Callers without restart argv may initialise an empty host
//! only and are refused transactionally if any model, scheduler, GPU, or prior
//! argv is present.
//!
//! Shutdown has a different terminal condition: it stops admission, releases
//! the request sender, and waits to a fixed deadline for the scheduler. It
//! joins only after `JoinHandle::is_finished()` proves the join non-blocking.
//! A missed deadline uses non-unwinding process exit while the handle and GPU
//! remain owned; the supervisor/OS, not Rust destructors racing a live kernel,
//! reclaims the context.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;

use super::model_host::ModelHost;
use super::serve_load::{Carried, load_model};
use crate::cli;

/// What a swap needs to know to undo itself.
#[derive(Debug)]
pub(crate) struct SwapOutcome {
    /// The argv of the model that was replaced, for a restore offer.
    pub previous: Option<cli::ServeArgs>,
}

/// Do not start a multi-minute load into a process that is on its way out.
///
/// The accept loop stops the moment shutdown is requested, so the model would
/// finish loading with nothing left to serve it — and the release it performs
/// first would take the OUTGOING model down with it, turning a clean drain into
/// an abrupt one.
///
/// Takes the answer rather than reading the global, so the rule is testable
/// without a test mutating a process-wide latch that has no reset and that
/// every other test calling `swap` would then trip over.
fn refuse_if_shutting_down(shutting_down: bool) -> Result<()> {
    anyhow::ensure!(
        !shutting_down,
        "shutdown is in progress — not starting a model load"
    );
    Ok(())
}

fn admit_in_process_initial_load(
    has_previous_args: bool,
    has_model: bool,
    has_scheduler: bool,
    has_gpu: bool,
) -> Result<()> {
    anyhow::ensure!(
        !has_previous_args && !has_model && !has_scheduler && !has_gpu,
        "live in-process model swap is disabled: exact restart argv is required; the running model is untouched"
    );
    Ok(())
}

fn refuse_if_cancelled(cancel: Option<&AtomicBool>) -> Result<()> {
    anyhow::ensure!(
        !cancel.is_some_and(|flag| flag.load(Ordering::Acquire)),
        "model launch preflight was cancelled; the running model is untouched"
    );
    Ok(())
}

fn acquire_swap_guard<'a>(
    host: &'a ModelHost,
    cancel: Option<&AtomicBool>,
) -> Result<parking_lot::MutexGuard<'a, ()>> {
    let Some(cancel) = cancel else {
        return Ok(host.swap_guard());
    };
    loop {
        refuse_if_cancelled(Some(cancel))?;
        if let Some(guard) = host.try_swap_guard_for(std::time::Duration::from_millis(50)) {
            return Ok(guard);
        }
    }
}

/// Refuse a model this binary has no kernels for, BEFORE anything is released.
///
/// The check itself already exists inside the load — but it runs at phase 3,
/// long after the outgoing model has been torn down. Discovering it there costs
/// a live server its model for a reason that was knowable from a JSON file
/// before anything was touched.
/// Returns the checkpoint's `model_type` so the caller can look up its launch profile
/// without reading config.json twice.
fn preflight_kernel_target(args: &cli::ServeArgs) -> Result<String> {
    let model_dir = super::serve_phases::resolve_model_dir(args)?;
    let (config, _) = super::serve_phases::load_model_config(&model_dir)?;
    if atlas_kernels::ptx_for_config(&config.model_type, config.hidden_size).is_none() {
        anyhow::bail!(
            "this build has no compiled kernels for model_type '{}' / hidden_size={} \
             (available: {:?}) — the running model is untouched",
            config.model_type,
            config.hidden_size,
            atlas_kernels::available_targets()
                .iter()
                .map(|t| &t.target.model)
                .collect::<Vec<_>>()
        );
    }
    Ok(config.model_type)
}

/// Copy the flags that describe the PROCESS, not the model, from the argv that
/// is running onto the argv that is about to.
///
/// A recipe describes a model: its checkpoint, quantization, context, batch
/// shape. It has no business deciding which socket the operator bound. The
/// socket is the starkest case — it is bound for the process lifetime and
/// cannot move, so a recipe's port is unserveable by construction.
fn carry_process_flags(next: &mut cli::ServeArgs, previous: &cli::ServeArgs) {
    next.bind = previous.bind.clone();
    next.port = previous.port;
    // Request dumping is an operator's observability choice, and no recipe sets
    // it. Without this, a swap replaces argv with the recipe's and the dump
    // silently stops — the file stays where it was, simply never written to
    // again, which is the worst way for a diagnostic to fail.
    next.dump = previous.dump.clone();
    // Auth is NOT carried here, because it is not carried through argv at all:
    // `swap` hands `load_model` the policy the host has held since boot, so a
    // recipe cannot drop `--require-auth` however its flags are written.
}

/// How long in-flight requests get to release the outgoing model before the
/// swap gives up and puts it back.
const DRAIN_GRACE: std::time::Duration = std::time::Duration::from_secs(30);
const SCHEDULER_RETIRE_GRACE: std::time::Duration = std::time::Duration::from_secs(30);

/// Block until `state` has no other owners, returning how many remain.
///
/// Split out from `release_state` so the waiting rule is testable without
/// standing up an `AppState`, which needs a loaded model.
fn wait_for_sole_owner<T>(state: &Arc<T>, grace: std::time::Duration) -> usize {
    let deadline = std::time::Instant::now() + grace;
    while Arc::strong_count(state) > 1 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    Arc::strong_count(state) - 1
}

/// Take the outgoing model out of the host and drop it, carrying the
/// process-scoped stores forward.
///
/// The wait is the point. An `Arc<AppState>` that outlives this window keeps
/// `request_tx` open, so the scheduler never drains and the join below never
/// returns — the swap wedges the server with no model loaded and no way back.
/// Two kinds of holder reach this point and the wait separates them: an
/// in-flight request finishes on its own, while a structural leak never does
/// and is reported as a refusal, with the model put back and still serving.
///
/// This is the reason the router is stated on the `ModelHost` rather than on an
/// `Arc<AppState>`: a state bound into a router layer is held for the router's
/// lifetime, so it could never reach a strong count of 1 and this function
/// could only ever refuse.
fn release_state(host: &Arc<ModelHost>, grace: std::time::Duration) -> Result<Carried> {
    // Nothing loaded. Nothing to drain, nothing to lose. NOT `Carried::
    // from_env()` on this branch: rebuilding them here is the silent
    // store-loss this whole type exists to prevent.
    let Some(state) = host.take() else {
        return host
            .process()
            .ok_or_else(|| anyhow::anyhow!("process-scoped state was never installed"));
    };
    // Taken off the outgoing state in the same breath as it leaves the host, so
    // there is no window in which these could be rebuilt by mistake.
    let carried = Carried::from_previous(&state);
    let holders = wait_for_sole_owner(&state, grace);
    if holders > 0 {
        // Transactional: the host had it a moment ago and nothing has been
        // freed, so putting it back restores the exact state we started in.
        host.publish(state);
        anyhow::bail!(
            "cannot swap: {holders} reference(s) to the running model outlived \
             the {}s drain window, so it can never be released. The model is \
             still serving. This is a leaked `Arc<AppState>` — most likely one \
             bound into a router layer or a spawned task rather than resolved \
             per request.",
            grace.as_secs()
        );
    }
    drop(state);
    Ok(carried)
}

/// Release the serving model and JOIN its scheduler before the process exits.
///
/// Returning from `main` while the scheduler thread is still dropping the model
/// races process teardown: the CUDA driver deinitialises underneath the drop,
/// `TransformerModel::drop`'s stream sync fails with CUDA_ERROR_DEINITIALIZED
/// and aborts, and every clean Ctrl+C dumps core. Same two steps a swap takes
/// before loading (release the state that owns `request_tx`, then join), under
/// the same guard so a swap in flight is not torn in half.
pub(crate) fn retire_for_shutdown(host: &Arc<ModelHost>) -> Result<()> {
    retire_for_shutdown_with(host, crate::scheduler::retirement::begin)
}

fn retire_for_shutdown_with(host: &Arc<ModelHost>, begin_retirement: impl FnOnce()) -> Result<()> {
    let Some(_swapping) = host.try_swap_guard_for(DRAIN_GRACE) else {
        super::scheduler_retire::fail_stop(
            "shutdown could not acquire the model lifecycle guard within 30s; restart required",
        );
    };
    // The HTTP drain window is over: begin scheduler retirement BEFORE waiting
    // for `Arc<AppState>` sole ownership. Active streaming/blocking handlers
    // hold `Arc<AppState>` until the scheduler stops their in-flight prefill
    // chunk or decode token and closes their sink; signaling retirement first
    // breaks the circular wait where `release_state` waited on handlers that
    // were waiting on the scheduler. One GPU chunk is still not preemptible,
    // so the grace must cover the longest one.
    begin_retirement();
    if let Err(error) = release_state(host, DRAIN_GRACE) {
        super::scheduler_retire::fail_stop(&format!(
            "shutdown could not release the serving state: {error:#}; restart required"
        ));
    }
    let Some(handle) = host.take_scheduler() else {
        if host.has_gpu_backend() {
            super::scheduler_retire::fail_stop(
                "GPU state is resident but its scheduler handle is missing; safe retirement cannot be proven",
            );
        }
        return Ok(());
    };
    let grace = super::scheduler_retire::retire_grace(SCHEDULER_RETIRE_GRACE);
    if !super::scheduler_retire::wait_until_finished(&handle, grace) {
        // Keep the proof-bearing handle owned until fail-stop.  Dropping it
        // would detach a thread that may still touch the model/GPU.
        host.set_scheduler(handle);
        super::scheduler_retire::fail_stop(&format!(
            "scheduler did not retire within {}s (one in-flight GPU chunk can exceed it; raise {}); leaving GPU state quarantined and exiting for supervisor restart",
            grace.as_secs(),
            super::scheduler_retire::RETIRE_GRACE_ENV,
        ));
    }
    match super::scheduler_retire::join_finished(handle) {
        Ok(result) => result,
        Err(handle) => {
            host.set_scheduler(handle);
            super::scheduler_retire::fail_stop(
                "scheduler completion proof was lost before join; leaving GPU state quarantined and exiting for supervisor restart",
            )
        }
    }
}

/// Mark the router/listening phases done and announce ready.
///
/// The listener is bound at boot and a swap never touches it, so `load_model`
/// cannot emit these itself — but a dashboard that never sees them keeps a
/// half-finished checklist and a LOADING pill over a server that is serving.
/// Both the swap and the restore need it, which is why it is a function and not
/// two copies.
fn signal_listener_phases(host: &Arc<ModelHost>) {
    if let Some((bind, port)) = host.bound() {
        spark_runtime::progress::phase(10, "router");
        spark_runtime::progress::phase(11, "listening");
        spark_runtime::progress::ready(port);
        tracing::info!(
            "swap complete: serving {} on {bind}:{port}",
            host.live_model().as_deref().unwrap_or("<unknown>"),
        );
    }
}

/// Replace the running model with the one `next` describes.
///
/// Blocking — it loads a model. Call it off the runtime.
pub(crate) fn swap(host: &Arc<ModelHost>, next: cli::ServeArgs) -> Result<SwapOutcome> {
    swap_with_profile(host, next, None, None)
}

/// [`swap`], able to re-exec: `recipe_argv` is the `spark serve` argv `next`
/// was parsed from. Every live recipe change uses this process boundary after
/// prospective validation and before teardown. Without exact restart argv a
/// non-empty host is refused while the running model remains untouched.
pub(crate) fn swap_with(
    host: &Arc<ModelHost>,
    next: cli::ServeArgs,
    recipe_argv: Option<Vec<String>>,
) -> Result<SwapOutcome> {
    swap_with_profile(host, next, recipe_argv, None)
}

/// [`swap_with`] with the exact selected built-in's launch profile.
///
/// Several named recipes intentionally target the same checkpoint but carry
/// different measured environments.  Passing the selected profile keeps that
/// identity through the TUI instead of resolving by model basename and taking
/// whichever recipe happens to appear first.
pub(crate) fn swap_with_profile(
    host: &Arc<ModelHost>,
    next: cli::ServeArgs,
    recipe_argv: Option<Vec<String>>,
    selected_profile: Option<super::model_profile::ModelProfile>,
) -> Result<SwapOutcome> {
    swap_with_profile_cancel(host, next, recipe_argv, selected_profile, None)
}

pub(crate) fn swap_with_profile_cancel(
    host: &Arc<ModelHost>,
    next: cli::ServeArgs,
    recipe_argv: Option<Vec<String>>,
    selected_profile: Option<super::model_profile::ModelProfile>,
    cancel: Option<&AtomicBool>,
) -> Result<SwapOutcome> {
    // Serialised HERE, not at one call site. Two swaps at once both call
    // `ModelHost::take`; the second gets `None`, mistakes it for a modelless
    // boot, and loads a second model onto a GPU that is already loading one.
    // A dashboard user pressing `s` twice is enough to reach that.
    // A recipe governs exactly one model for the process lifetime.
    crate::recipe_contract::resolved::refuse_swap()?;
    let _swapping = acquire_swap_guard(host, cancel)?;
    refuse_if_cancelled(cancel)?;

    // Taken from the host, not a parameter: a caller that has to supply the
    // outgoing argv is a caller that can forget, and forgetting it silently
    // disables restore-on-failure.
    let previous_args = host.args();

    let mut next = next;
    if let Some(previous) = previous_args.as_ref() {
        if next.port != previous.port || next.bind != previous.bind {
            tracing::warn!(
                "this recipe asks to bind {}:{}, but the listener is on {}:{} for the \
                 process lifetime — serving the new model there instead",
                next.bind,
                next.port,
                previous.bind,
                previous.port
            );
        }
        carry_process_flags(&mut next, previous);
    }

    // Whoever waited on the guard may be asking for what the winner just
    // loaded. Re-checked AFTER the guard, not before: while a request waited,
    // the winner may have loaded exactly the model it wanted, and repeating a
    // multi-minute load is a second outage for nothing.
    //
    // Compared as a WHOLE argv rather than by model id, because switching
    // between two recipes for the SAME checkpoint is a real swap. And AFTER
    // carrying, not before: `previous_args` is the LIVE argv and already holds
    // the carried flags while `next` does not, so comparing them first would
    // make the two unequal whenever any process flag was set — which is exactly
    // when requests queue behind a swap.
    if previous_args.as_ref() == Some(&next) && host.current().is_some() {
        tracing::info!("swap: the requested model is already the one serving — nothing to do");
        return Ok(SwapOutcome {
            previous: previous_args,
        });
    }

    // Refuse before anything is torn down. A bad flag combination, a missing
    // checkpoint or an impossible VRAM budget must cost nothing — the window
    // where the server has no model is opened only for a config that has
    // already passed everything cheap.
    cli::validate_serve_args(&next).map_err(|e| anyhow::anyhow!("{e}"))?;

    refuse_if_shutting_down(crate::tui::shutdown::requested())?;
    refuse_if_cancelled(cancel)?;

    // The load spawns Tokio tasks. Entering here rather than at each call site
    // means a caller that is already inside the runtime and one that is not
    // (the TUI's plain thread) both work, and neither has to know which it is.
    let runtime = host.runtime();
    let _entered = runtime.as_ref().map(|h| h.enter());

    // Multi-rank is out of scope and must fail loudly rather than half-swap:
    // the EP worker takes the model by `Option::take` and only returns when the
    // head exits, so there is no "load a different model" command to send it.
    //
    // UNTESTED ON HARDWARE, and unavoidably so — multi-node needs two boxes and
    // a real EP deployment. What is verified is that the refusal fires; what is
    // not is the behaviour of a head or worker that reaches this path in a
    // genuine world_size > 1 run.
    anyhow::ensure!(
        next.world_size <= 1 && next.rank == 0,
        "hot-swap is single-node only (world_size={}, rank={})",
        next.world_size,
        next.rank
    );

    // Cheapest checks first: this one reads the checkpoint's config.json, so it
    // runs after the ones that need nothing but the argv.
    let model_type = preflight_kernel_target(&next)?;
    refuse_if_cancelled(cancel)?;

    // A model with a launch profile (memory, run-alone, env) — DeepSeek-V4.1 today. Refused
    // HERE, before anything is released, only when it could never fit on this machine at all;
    // the real admission is after the release below, on the memory that is actually free.
    let profile = match selected_profile {
        Some(profile) => {
            anyhow::ensure!(
                profile.model_type == model_type,
                "{} launch profile declares model_type {:?}, but the selected checkpoint is {:?}",
                profile.recipe_id,
                profile.model_type,
                model_type
            );
            Some(profile)
        }
        None => super::model_profile::profile_for(
            next.model_from_path
                .as_deref()
                .and_then(|p| p.to_str())
                .or(next.model.as_deref()),
            &model_type,
        )?,
    };
    // A measured profile is an exact contract, not a bag of defaults. Validate the
    // environment the re-exec or initial load would actually see. In particular this
    // prevents an inherited cache/reserve override from silently turning the memory-safe
    // native-MTP profile into the memory-marginal high-prefill profile (or vice versa).
    if let Some(profile) = profile.as_ref()
        && let Some(receipt) = profile.validate_environment_with(
            |name| {
                let operator = super::swap_env::operator_environment_value(name);
                super::model_profile::unicode_environment_value(name, operator)
                    .map(|value| value.or_else(|| profile.env.get(name).cloned()))
            },
            option_env!("ATLAS_RELEASE_PROFILE_TRACK"),
        )?
    {
        super::model_profile::log_receipt(&receipt);
    }
    // Validate the environment the next load would see before opening the
    // modelless window. Use `operator_environment_value` so profile-owned keys
    // from the currently loaded model (which re-exec clears) do not pollute the
    // prospective check for the next model.
    cli::validate_dflash_runtime_environment_with(&next, |name| {
        super::swap_env::operator_environment_value(name)
            .map(|value| {
                value
                    .into_string()
                    .unwrap_or_else(|_| "<non-Unicode>".to_string())
            })
            .or_else(|| {
                profile
                    .as_ref()
                    .and_then(|candidate| candidate.env.get(name).cloned())
            })
    })
    .map_err(anyhow::Error::msg)?;
    super::flashinfer_artifact::preflight_prospective(&model_type, profile.as_ref())
        .map_err(anyhow::Error::msg)?;
    refuse_if_cancelled(cancel)?;
    if let Some(p) = profile.as_ref().filter(|p| p.run_alone) {
        let total = super::model_profile::mem_total_bytes()?;
        let scope = if spark_runtime::startup_ledger::is_discrete_device() {
            "this GPU"
        } else {
            "this machine"
        };
        anyhow::ensure!(
            total >= p.required_bytes(false),
            "{} needs {:.1} GB free but {scope} has {:.1} GB in total — the running model \
             is untouched",
            p.recipe_id,
            p.required_bytes(false) as f64 / 1e9,
            total as f64 / 1e9
        );
    }

    // A recipe launch always crosses a process boundary.  Even an apparently
    // compatible environment cannot make an in-process teardown safe when a
    // scheduler or PLE worker fails to retire: POSIX exec replaces every old
    // thread and the CUDA driver context without freeing allocations under a
    // live kernel.  Crucially this happens BEFORE `release_state`, so an exec
    // error leaves the old model serving.
    if let Some(argv) = recipe_argv.as_ref() {
        refuse_if_cancelled(cancel)?;
        anyhow::ensure!(
            super::swap_env::reexec_enabled(),
            "recipe launch requires a clean process restart; re-exec is disabled (ATLAS_SWAP_REEXEC=0), so the running model is untouched"
        );
        let empty = std::collections::BTreeMap::new();
        tracing::warn!("swap: validated recipe will replace the process before model teardown");
        return Err(super::swap_env::exec(
            argv,
            profile.as_ref().map_or(&empty, |p| &p.env),
        ));
    }

    // A caller without exact restart argv may initialise an empty host, but it
    // may never retire live state in-process.  Refuse transactionally rather
    // than reintroducing an unbounded join or detaching a GPU owner.
    admit_in_process_initial_load(
        previous_args.is_some(),
        host.current().is_some(),
        host.has_scheduler(),
        host.has_gpu_backend(),
    )?;

    // The policy the host has held since boot. NOT rebuilt from `next`: a
    // recipe's argv must not be able to drop `--require-auth` from a server
    // that was started with it.
    let auth = host.auth();

    // Empty-host initialisation only: the live-host case returned through
    // pre-teardown exec or was refused above.
    let carried = release_state(host, DRAIN_GRACE)?;

    // Load the first model into the empty host.
    let next_args = next.clone();
    // Forward the dashboard's run-handle channel so the reloaded scheduler
    // republishes its levers and stats to the TUI.
    let tui_handles_tx = host.tui_handles();
    // Launch profile: verify its environment is already present (never mutate
    // process env once Tokio/TUI threads are running), then run-alone admission
    // on currently free memory.
    let admitted = match profile.as_ref() {
        None => Ok(()),
        Some(p) => super::model_profile::ensure_env_present(p).and_then(|()| {
            if let Some(receipt) = p.validate_environment_with(
                |name| {
                    super::model_profile::unicode_environment_value(name, std::env::var_os(name))
                },
                option_env!("ATLAS_RELEASE_PROFILE_TRACK"),
            )? {
                super::model_profile::log_receipt(&receipt);
            }
            let dspark_on = std::env::var("ATLAS_DSV41_DSPARK")
                .ok()
                .or_else(|| p.env.get("ATLAS_DSV41_DSPARK").cloned())
                .is_some_and(|v| v == "1");
            let is_discrete = spark_runtime::startup_ledger::is_discrete_device();
            super::model_profile::mem_available_bytes().and_then(|available| {
                super::model_profile::admit_for_device(p, available, dspark_on, is_discrete)
            })
        }),
    };
    let load_result = match admitted {
        Err(e) => Err(e),
        Ok(()) => load_model(next, tui_handles_tx, carried, auth),
    };
    let load_err = match load_result {
        Ok(Some(prepared)) => {
            host.set_scheduler(prepared.scheduler);
            host.set_gpu_backend(prepared.gpu);
            host.set_oom_watchdog(prepared.oom_watchdog);
            host.set_args(next_args);
            host.publish(prepared.state);
            signal_listener_phases(host);
            return Ok(SwapOutcome {
                previous: previous_args,
            });
        }
        Ok(None) => anyhow::anyhow!("hot-swap reached an EP-worker path on rank 0"),
        Err(e) => e,
    };

    // `admit_in_process_initial_load` guarantees this path only runs on an
    // empty host (`previous_args.is_none()`), so there is no outgoing model to
    // restore; live recipe swaps always cross `swap_env::exec` before teardown.
    Err(load_err.context("the new model failed to load into the empty host"))
}

#[cfg(test)]
#[path = "model_swap_tests.rs"]
mod tests;

#[cfg(test)]
mod shutdown_retirement_tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::wait_for_sole_owner;

    #[test]
    fn retirement_is_signaled_before_releasing_serving_state() {
        let source = include_str!("model_swap.rs");
        let retire_fn = source
            .split("pub(crate) fn retire_for_shutdown(")
            .nth(1)
            .expect("retire_for_shutdown definition")
            .split("fn signal_listener_phases(")
            .next()
            .expect("end of retire_for_shutdown");
        assert!(
            retire_fn.contains("crate::scheduler::retirement::begin"),
            "retire_for_shutdown must wire scheduler retirement"
        );
        let begin_pos = retire_fn
            .find("begin_retirement();")
            .expect("begin_retirement call");
        let release_pos = retire_fn
            .find("release_state(host, DRAIN_GRACE)")
            .expect("release_state call");
        assert!(
            begin_pos < release_pos,
            "scheduler retirement must begin before waiting for Arc<AppState> sole ownership"
        );
    }

    #[test]
    fn signaling_retirement_first_breaks_stream_holder_deadlock() {
        let state = Arc::new(42u32);
        let retired = Arc::new(AtomicBool::new(false));
        let holder_state = Arc::clone(&state);
        let holder_retired = Arc::clone(&retired);
        let handle = std::thread::spawn(move || {
            while !holder_retired.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(5));
            }
            drop(holder_state);
        });

        // Before retirement is signaled, the active stream holder keeps Arc alive.
        assert_eq!(wait_for_sole_owner(&state, Duration::from_millis(40)), 1);

        // Once retirement begins before release_state waits, the holder exits and drops Arc.
        retired.store(true, Ordering::Release);
        assert_eq!(wait_for_sole_owner(&state, Duration::from_secs(2)), 0);
        assert!(super::super::scheduler_retire::wait_until_finished(
            &handle,
            Duration::from_secs(2),
        ));
        super::super::scheduler_retire::join_finished(handle)
            .expect("holder thread finished")
            .expect("holder thread did not panic");
    }
}
