// SPDX-License-Identifier: AGPL-3.0-only

// Warnings are enforced as errors in CI via `cargo clippy -- -D warnings`.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

//! Flux Engine (`flux-server` / `spark`) — High-Velocity Blackwell SM121a Rust LLM Inference Server (derived from Atlas).
//!
//! Startup sequence:
//! 1. Parse CLI args
//! 2. Load model config
//! 3. Initialize GPU backend (AtlasCudaBackend)
//! 4. Load model weights (SafetensorsLoader)
//! 5. Build model via factory
//! 6. Load tokenizer
//! 7. Spawn scheduler thread
//! 8. Start axum HTTP server

// Global allocator: mimalloc. Faster than glibc's ptmalloc on the
// Vec-heavy / String-heavy allocations in the per-token scheduler
// hot path (draft Vecs, D2H buffers, content sanitiser, accumulator
// strings). Picked over jemalloc for: smaller binary, better aarch64
// performance, and active maintenance.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod adaptive_sampler;
mod anthropic;
mod api;
mod atlas_home;
mod auth;
mod citation;
mod cli;
mod conversation_store;
mod debug_log;
mod display_carveout;
pub mod dsv41;
mod error_hints;
pub mod grammar;
mod halluc_probe;
mod hint_injector;
mod llmlingua;
mod load_debug;
mod lookback_lens;
mod loop_detector;
mod loop_simhash;
mod lqer;
mod main_modules;
pub mod metrics;
mod model_download;
mod model_resolver;
mod moe_quality;
mod ngram;
mod observation_mask;
mod openai;
mod prefill_twophase_gate;
pub(crate) mod pyjson;
mod rate_limiter;
pub mod reasoning_parser;
pub mod recipe;
pub mod recipe_contract;
mod refusal;
mod request_dumper;
mod response_store;
mod runtime_settings;
// REST retrieval draft store: conditional pre-emption of the DFlash
// proposal in `scheduler::mtp_step`. Inert unless ATLAS_REST_STORE is set.
mod rest_store;
mod retrieval_heads;
mod scheduler;
mod scheduling_policy;
mod session_manager;
mod symbol_trie;
mod task_pin;
mod term_color;
#[cfg(test)]
mod test_paths;
mod tokenizer;
mod tool_arg_dedup;
pub mod tool_parser;
mod tool_rag;
mod tool_salvage;
mod tscg;
pub mod tui;

use anyhow::Result;
use clap::Parser;
use std::ffi::{OsStr, OsString};

use crate::cli::{Cli, Command, EntryCli, EntryCommand};
use crate::main_modules::serve;

pub(crate) use crate::main_modules::AppState;

/// Re-export for convenience in api.rs / anthropic.rs.
pub type ModelBehavior = atlas_kernels::ModelBehavior;

fn flux_env_alias_key(key: &OsStr) -> Option<String> {
    const META_KEYS: &[&str] = &[
        "FLUX_MODE",
        "FLUX_QUIET",
        "FLUX_MODELS_ROOT",
        "FLUX_ATLAS_ROOT",
        "FLUX_ACTIVE_ARGV_FILE",
        "FLUX_OPT_OUT",
        "FLUX_THINKING",
        // Read by the server under their FLUX_ names only.
        "FLUX_LOAD_DEBUG",
        "FLUX_LOAD_DEBUG_FILE",
        "FLUX_DEBUG",
        "FLUX_DEBUG_FILE",
        "FLUX_DEBUG_MAX_MB",
        "FLUX_DEBUG_FILTER",
    ];
    let key = key.to_str()?;
    let suffix = key.strip_prefix("FLUX_")?;
    if META_KEYS.contains(&key) || suffix.is_empty() {
        return None;
    }
    Some(match suffix {
        "QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB" => {
            "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB".to_string()
        }
        "PREFILL_WIDE" => "ATLAS_QWEN4_PREFILL_WIDE".to_string(),
        _ => format!("ATLAS_{suffix}"),
    })
}

fn flux_env_alias_if_absent(
    key: &OsStr,
    value: OsString,
    atlas_key_exists: impl FnOnce(&str) -> bool,
) -> Option<(String, OsString)> {
    let atlas_key = flux_env_alias_key(key)?;
    if atlas_key_exists(&atlas_key) {
        return None;
    }
    Some((atlas_key, value))
}

/// Synchronize aliases before constructing the asynchronous runtime. Values stay
/// raw so each relevant selector's strict parser can diagnose non-Unicode input.
fn sync_flux_env_aliases() {
    let vars: Vec<(OsString, OsString)> = std::env::vars_os().collect();
    for (key, value) in vars {
        if let Some((atlas_key, value)) =
            flux_env_alias_if_absent(&key, value, |alias| std::env::var_os(alias).is_some())
        {
            // SAFETY: synchronous `main` calls this before the runtime or any worker threads start.
            unsafe {
                std::env::set_var(&atlas_key, &value);
            }
        }
    }
}

/// Export the engine controls implied by `FLUX_DFLASH_LOOKUP` (default off,
/// so this sets nothing unless an operator opted in).
fn sync_dflash_lookup_controls() {
    let implied = cli::dflash_lookup::implied_controls(cli::dflash_environment_value);
    for (name, value) in implied {
        // SAFETY: synchronous `main` calls this before the runtime or any worker threads start.
        unsafe {
            std::env::set_var(name, value);
        }
    }
}

fn main() -> Result<()> {
    sync_flux_env_aliases();
    sync_dflash_lookup_controls();
    // Parse CLI before starting the Tokio runtime so process-wide env setup
    // (`--profile`, FlashInfer non-aarch64 gate bypass) runs while single-threaded.
    let entry = EntryCli::parse();
    if let EntryCommand::Serve(ref args) = entry.command {
        if args.profile {
            // SAFETY: synchronous `main` calls this before Tokio or any worker threads start.
            unsafe {
                std::env::set_var("ATLAS_PROFILE", "1");
            }
        }
        // SAFETY: synchronous `main` calls this before Tokio or any worker threads start.
        unsafe {
            main_modules::flashinfer_artifact::disable_on_unsupported_arch();
        }
    }
    run_application(entry)
}

#[tokio::main]
async fn run_application(entry: EntryCli) -> Result<()> {
    // EntryCli was parsed before subscriber install so the TUI gate can see `--no-tui`.
    // clap emits no tracing events, so plain-mode output is unchanged.
    let cli = match entry.command {
        EntryCommand::DisplayCarveout(args) => {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| "info".into()),
                )
                .init();
            return display_carveout::run(args);
        }
        EntryCommand::Serve(args) => Cli {
            command: Command::Serve(args),
        },
    };
    let (no_tui, rank) = match &cli.command {
        Command::Serve(args) => (args.no_tui || args.rank > 0, args.rank),
    };

    let plain_mode = tui::plain_mode(no_tui);
    tui::set_active(!plain_mode);
    let load_capture = load_debug::install(rank as u32);
    let debug_file = debug_log::install(rank as u32);
    let tui_channels = if plain_mode && load_capture.is_none() && !debug_file {
        // The pre-TUI init, byte-for-byte: this exact fmt layout is the
        // contract every benchmark driver and gate script greps.
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .init();
        None
    } else if plain_mode {
        // The same console output as above (byte-identical, see
        // `debug_log` tests), now behind a per-layer filter so the load
        // capture and the debug file beside it can see debug while stdout
        // does not.
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        tracing_subscriber::registry()
            .with(debug_log::console_layer(
                std::io::stdout,
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            ))
            .with(load_debug::layer())
            .with(debug_log::layer())
            .init();
        None
    } else {
        let (progress_tx, progress_rx) = std::sync::mpsc::channel();
        tui::init::install_tty_subscriber(progress_tx);
        Some(progress_rx)
    };

    // Race the server against shutdown. No spawn: `serve()` is a real future that
    // yields while its blocking startup runs on the blocking pool, so pinning it
    // here is enough for `select!` to poll the other branch. (It would NOT be
    // enough if startup still blocked inside the future — `select!` chooses at
    // await points, it does not preempt.)
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<&'static str>();
    tui::shutdown::arm_startup_escape(shutdown_tx);
    let result = match cli.command {
        Command::Serve(args) => {
            let serving = serve(args, tui_channels);
            tokio::pin!(serving);
            // Only a SEND means shutdown. The sender is parked for the life of the
            // process rather than dropped when startup ends, so the channel should
            // never close; this arm exists so that if one ever did, a closed
            // channel could not masquerade as a shutdown and kill a healthy server.
            let shutdown_signal = async {
                match shutdown_rx.await {
                    Ok(reason) => reason,
                    Err(_) => std::future::pending::<&'static str>().await,
                }
            };
            tokio::pin!(shutdown_signal);
            tokio::select! {
                res = &mut serving => res,
                reason = &mut shutdown_signal => {
                    // Cancelled before the server came up. Nothing is in flight
                    // and no client is connected, so there is nothing to drain —
                    // the startup task is abandoned where it stands.
                    tracing::info!(
                        "Shutdown requested ({reason}) during startup — exiting before the server came up"
                    );
                    // Cleanup that would otherwise run below, then exit without
                    // waiting on the runtime: a task parked inside a synchronous
                    // CUDA call cannot be aborted, and dropping the runtime would
                    // block on it — reintroducing the very wait this fixes.
                    tui::stop_and_join(std::time::Duration::from_secs(2));
                    tui::terminal_guard::restore();
                    tui::init::flush_tee();
                    std::process::exit(0);
                }
            }
        }
    };
    // If serve() returned while the TUI owned the screen (startup error, clean
    // shutdown), stop the dashboard thread and wait for its TerminalGuard to
    // drop BEFORE the error prints — main's exit never runs another thread's
    // Drop, and a bare restore() races the thread's raw-mode entry when
    // serve() fails within milliseconds. restore() stays as the backstop.
    tui::stop_and_join(std::time::Duration::from_secs(2));
    tui::terminal_guard::restore();
    tui::init::flush_tee();
    if result.is_err() {
        // Before `main` prints the error chain, so the cause stays last.
        load_debug::report_load_failure();
    }
    result
}

#[cfg(test)]
mod flux_env_alias_tests {
    use super::{flux_env_alias_if_absent, flux_env_alias_key};
    use std::ffi::{OsStr, OsString};

    #[test]
    fn meta_empty_and_unrelated_keys_do_not_alias() {
        for key in [
            "FLUX_MODE",
            "FLUX_QUIET",
            "FLUX_MODELS_ROOT",
            "FLUX_ATLAS_ROOT",
            "FLUX_ACTIVE_ARGV_FILE",
            "FLUX_OPT_OUT",
            "FLUX_THINKING",
            "FLUX_LOAD_DEBUG",
            "FLUX_DEBUG",
            "FLUX_DEBUG_FILTER",
            "FLUX_",
            "OTHER_VALUE",
            "ATLAS_MAX_BATCH_TOKENS",
        ] {
            assert_eq!(flux_env_alias_key(OsStr::new(key)), None, "{key}");
        }
    }

    #[test]
    fn ordinary_and_special_alias_names_are_preserved() {
        for (key, expected) in [
            ("FLUX_MAX_BATCH_TOKENS", "ATLAS_MAX_BATCH_TOKENS"),
            ("FLUX_PREFILL_WIDE", "ATLAS_QWEN4_PREFILL_WIDE"),
            (
                "FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB",
                "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB",
            ),
        ] {
            assert_eq!(
                flux_env_alias_key(OsStr::new(key)).as_deref(),
                Some(expected)
            );
        }
    }

    #[test]
    fn existing_atlas_values_win_even_when_empty() {
        for value in ["", "4096"] {
            assert_eq!(
                flux_env_alias_if_absent(
                    OsStr::new("FLUX_MAX_BATCH_TOKENS"),
                    OsString::from(value),
                    |key| key == "ATLAS_MAX_BATCH_TOKENS",
                ),
                None,
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_keys_are_ignored_and_alias_values_are_preserved_raw() {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};

        assert_eq!(flux_env_alias_key(OsStr::from_bytes(b"FLUX_\xff")), None);
        let raw = OsString::from_vec(b"\xff".to_vec());
        assert_eq!(
            flux_env_alias_if_absent(OsStr::new("UNRELATED"), raw.clone(), |_| false),
            None,
        );
        assert_eq!(
            flux_env_alias_if_absent(OsStr::new("FLUX_MAX_BATCH_TOKENS"), raw.clone(), |_| false,),
            Some(("ATLAS_MAX_BATCH_TOKENS".to_string(), raw.clone())),
        );
        assert_eq!(
            flux_env_alias_if_absent(OsStr::new("FLUX_MAX_BATCH_TOKENS"), raw, |_| true),
            None,
        );
    }
}
