// SPDX-License-Identifier: AGPL-3.0-only

//! Loading one model, and the process-scoped state that outlives every model.
//!
//! This is the boundary a swap re-runs from. Everything ABOVE it — the banner,
//! the signal listeners, the dashboard thread, and the `--profile` env setup —
//! happens once per PROCESS and lives in `main` / `serve::startup`. Everything
//! here is model-dependent and is re-executed, unchanged, for every model load
//! (with the UMA OOM watchdog scoped to each loaded GPU instance via
//! [`OomWatchdogGuard`]).
//!
//! Splitting it this way keeps process-scoped initialization (signal handlers,
//! dashboard thread, HTTP listener) distinct from model-scoped state.
//!
//! [`Carried`] is the other half of the contract. The three stores in it
//! outlive any model, and rebuilding them on a swap silently drops stored
//! conversations and responses and resets every rate-limit bucket — no error,
//! just a user noticing their history is gone. Keeping them in one struct and
//! taking that struct as a PARAMETER of [`load_model`] makes carrying them the
//! only way to load a second model: the compiler asks, so nobody has to
//! remember.

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::mpsc;

use crate::api::InferenceRequest;
use crate::cli;
use crate::main_modules::AppState;
use crate::main_modules::serve_phases;
use crate::tokenizer::ChatTokenizer;
use crate::{
    conversation_store, rate_limiter, response_store, scheduler, scheduling_policy, session_manager,
};

/// State that OUTLIVES any model and must be carried across a swap.
#[derive(Clone)]
pub(crate) struct Carried {
    pub response_store: Arc<response_store::ResponseStore>,
    pub rate_limiter: Arc<rate_limiter::RateLimiter>,
    pub conversation_store: Arc<conversation_store::ConversationStore>,
}

impl Carried {
    /// First boot: build them once, from the environment.
    ///
    /// Installed on the HOST before the listener binds, so everything
    /// process-scoped is reachable while no model is loaded — and so there is
    /// exactly one of each: handlers refund through the same limiter the
    /// middleware debits, and read the same stores a swap carries forward.
    ///
    /// # Errors
    /// Currently infallible; the `Result` signature reserves room for strict
    /// environment validation of rate-limit and store parameters so typos are
    /// rejected rather than silently defaulted.
    pub fn from_env() -> Result<Self, String> {
        Ok(Self {
            response_store: response_store::ResponseStore::from_env(),
            rate_limiter: rate_limiter::RateLimiter::from_env(),
            conversation_store: conversation_store::ConversationStore::from_env(),
        })
    }

    /// Carry process-scoped stores forward across a model swap so stored
    /// conversations, responses, and rate-limit buckets are preserved.
    pub fn from_previous(previous: &AppState) -> Self {
        Self {
            response_store: previous.response_store.clone(),
            rate_limiter: previous.rate_limiter.clone(),
            conversation_store: previous.conversation_store.clone(),
        }
    }
}

/// Abort-on-drop guard for the UMA OOM watchdog Tokio task.
///
/// Dropping a bare `tokio::task::JoinHandle` detaches the task; wrapping it
/// here guarantees the watchdog stops immediately on load error or shutdown.
pub(crate) struct OomWatchdogGuard(tokio::task::JoinHandle<()>);

impl Drop for OomWatchdogGuard {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// What one load hands back: a model ready to serve, and the thread serving it.
///
/// A struct rather than a tuple since the scheduler handle accompanies it — five
/// positional fields where two are `Arc`s and two are addresses is a swap
/// waiting to happen at the call site.
pub(crate) struct Prepared {
    pub state: Arc<AppState>,
    pub bind: String,
    pub port: u16,
    /// The scheduler thread. Terminal shutdown waits to a deadline, then joins
    /// only after completion is proven. Live recipe changes replace the
    /// process, because Rust cannot safely kill a stuck owner thread.
    /// Not an `Option`: a rank that produces a `Prepared` always has one, and
    /// the EP-worker rank that does not returns `Ok(None)` instead.
    pub scheduler: std::thread::JoinHandle<()>,
    /// The GPU backend instance this model was built and served with.
    ///
    /// Kept alongside `scheduler`, not just handed to the model, because a
    /// terminal shutdown must keep it owned until scheduler completion is
    /// proven. On a missed deadline the process exits without unwinding, so
    /// the OS reclaims the CUDA context rather than Rust destructors racing a
    /// live kernel.
    pub gpu: std::sync::Arc<dyn spark_runtime::gpu::GpuBackend>,
    /// Abort-on-drop UMA OOM watchdog guard (present only on CUDA UMA hosts).
    pub oom_watchdog: Option<OomWatchdogGuard>,
}

/// Report the admitted DFlash lookup selector mode once CLI environment
/// validation has succeeded.
fn log_dflash_lookup(dflash: bool) {
    use crate::cli::dflash_lookup;
    let Ok(mode) = dflash_lookup::lookup_mode(cli::dflash_environment_value) else {
        return;
    };
    if !dflash {
        if mode != dflash_lookup::DflashLookup::Off {
            tracing::warn!(
                "{}={} has no effect without --dflash; lookup drafting is not engaged",
                dflash_lookup::LOOKUP_ENV,
                mode.as_str()
            );
        }
        return;
    }
    if let Some(receipt) = dflash_lookup::engaged_receipt(mode) {
        tracing::info!("{receipt}");
    }
}

/// Load one model and start its scheduler.
///
/// Blocking, and long — tens of seconds of weight reading, KV allocation and
/// graph capture with not one `await` in it. Call it on the blocking pool or
/// from a plain thread, never on a runtime worker.
///
/// `Ok(None)` means this rank is an EP worker: it ran its command loop and has
/// nothing for the async tail to serve.
///
/// `carried` and `auth` are the process-scoped state, passed in rather than
/// rebuilt here, so a swap preserves stored conversations, responses, and
/// `--require-auth` policy.
///
/// `tui_handles_tx` is the dashboard's `RunHandles` sender, threaded through so
/// each load republishes the levers sampled by the Stats pane.
///
/// Note: `scheduler::run` gates the loop watchdog via the process-global
/// `scheduler::set_enable_loop_watchdog`; the published lever reflects the
/// initial state rather than dynamically toggling the scheduler loop.
pub(crate) fn load_model(
    mut args: cli::ServeArgs,
    tui_handles_tx: Option<std::sync::mpsc::Sender<crate::tui::RunHandles>>,
    carried: Carried,
    auth: Option<Arc<crate::auth::AuthConfig>>,
) -> Result<Option<Prepared>> {
    cli::validate_dflash_runtime_environment(&args).map_err(anyhow::Error::msg)?;
    if args.dflash {
        tracing::info!(
            "DFlash method contract: CLI and runtime are pinned to flat; DDTree is fail-closed"
        );
    }
    log_dflash_lookup(args.dflash);
    // Parse before model resolution/weight loading. Recognized dynamic values
    // are rejected against the resolved drafter family below; unknown values
    // are always an immediate user error rather than a fallback.
    let dspark_verify_mode = args
        .dspark_verify_mode
        .parse::<spark_model::weight_loader::DsparkVerifyMode>()
        .context("Invalid --dspark-verify-mode")?;

    // 0. Resolve model directory from HF ID or path
    spark_runtime::progress::phase(1, "model resolve");
    let model_dir = serve_phases::resolve_model_dir(&args)?;

    tracing::info!("Port: {}", args.port);

    tracing::info!("SSM decode dtype: f32 (full precision)");

    // 1. Load model config (supports HF config.json and Mistral params.json)
    spark_runtime::progress::phase(2, "config");
    let (mut config, config_json) = serve_phases::load_model_config(&model_dir)?;
    super::flashinfer_artifact::preflight_live(&config.model_type).map_err(anyhow::Error::msg)?;
    let deepseek_spec_admission = super::deepseek_spec_admission::admit(
        &config.model_type,
        args.dflash,
        args.speculative,
        std::env::var("ATLAS_DSV41_DSPARK").ok().as_deref(),
    )?;
    super::deepseek_spec_admission::log(deepseek_spec_admission);
    config.qwen4_qsa = args.qwen4_qsa;
    if args.qwen4_qsa && !config.is_qwen4_exp() {
        anyhow::bail!("--qwen4-qsa requires a qwen4_exp checkpoint");
    }
    if config.is_qwen4_exp() && args.max_seq_len > 2048 && !args.qwen4_qsa {
        anyhow::bail!(
            "Qwen4 contexts above 2048 require --qwen4-qsa; dense fallback is intentionally capped"
        );
    }
    if config.is_qwen4_exp() {
        if args.max_batch_size > 1
            && spark_model::layers::qwen4_verify_lanes::batched_k5_selected()?
        {
            spark_model::layers::qwen4_verify_lanes::require_fixed_k5_policy()?;
            if spark_model::layers::qwen4_verify_lanes::adaptive_k_selected()? {
                spark_model::layers::qwen4_verify_lanes::require_adaptive_k_policy()?;
            }
        }
        super::qwen4_concurrency::validate_qwen4_concurrency(
            super::qwen4_concurrency::Qwen4Concurrency {
                qsa: args.qwen4_qsa,
                max_batch_size: args.max_batch_size,
                kv_cache_dtype: &args.kv_cache_dtype,
                native_context: config.max_position_embeddings,
                max_seq_len: args.max_seq_len,
                speculative: args.speculative
                    || args.dflash
                    || args.self_speculative
                    || args.ngram_speculative,
                native_mtp_only: args.speculative
                    && !args.dflash
                    && !args.self_speculative
                    && !args.ngram_speculative,
                dflash_only: args.dflash && !args.self_speculative && !args.ngram_speculative,
                batched_k5: spark_model::layers::qwen4_verify_lanes::batched_k5_selected()?,
                batched_adaptive_k: spark_model::layers::qwen4_verify_lanes::adaptive_k_selected()?,
                num_drafts: args.num_drafts,
                swap_space_gb: args.swap_space_gb,
                high_speed_swap: args.high_speed_swap,
                prefix_caching: args.enable_prefix_caching,
                ssm_cache_slots: args.ssm_cache_slots,
                world_size: args.world_size,
                tp_size: args.tp_size,
                ep_size: args.ep_size,
                serial_k5: super::qwen4_concurrency::SerialK5Request {
                    gate: spark_model::model::qwen4_k5_cmulti::k5_cmulti_selected()
                        .map_err(anyhow::Error::msg)?,
                    native_mtp_only: args.speculative
                        && !args.dflash
                        && !args.self_speculative
                        && !args.ngram_speculative,
                    num_drafts: args.num_drafts,
                    // Same exact-"1" reads as the scheduler's selectors
                    // (`step_mtp` K2/K3 CSEQ and cross-sequence DFlash).
                    cross_request_verify: spark_model::layers::mtp_k2_batch_cseq_enabled()
                        || spark_model::layers::mtp_k3_batch_cseq_enabled()
                        || std::env::var("ATLAS_DFLASH_BATCHED_VERIFY").ok().as_deref()
                            == Some("1"),
                },
                kv_pool_tokens: args.kv_pool_tokens,
            },
        )
        .map_err(anyhow::Error::msg)?;
        // Native C2-C32 batching is the default (`ATLAS_QWEN4_NATIVE_BATCH`
        // unset); refuse its contradictions before the weights load.
        let selector = spark_model::qwen4_batch_policy::NativeBatchSelector::from_env()
            .map_err(anyhow::Error::msg)?;
        let facts = spark_model::qwen4_batch_policy::ProcessFacts::from_env(
            true,
            args.profile || std::env::var_os("ATLAS_PROFILE").is_some(),
            config.vocab_size,
            args.qwen4_qsa,
        );
        let qsa_rows = spark_model::qwen4_batch_policy::qsa_rows_requested_from_env()?;
        super::qwen4_concurrency::admit_qwen4_native_batch(
            selector,
            &facts,
            qsa_rows,
            args.max_batch_size,
        )
        .map_err(anyhow::Error::msg)?;
    }
    if config.is_qwen4_exp() && args.max_seq_len > config.max_position_embeddings {
        let factor = (args.max_seq_len as f32 / config.max_position_embeddings as f32).ceil();
        config.yarn_factor = factor;
        config.yarn_beta_fast = 32.0;
        config.yarn_beta_slow = 1.0;
        config.yarn_original_max_position_embeddings = config.max_position_embeddings;
        tracing::info!(
            factor,
            original_context = config.max_position_embeddings,
            requested_context = args.max_seq_len,
            "Qwen4 static YaRN long-context profile enabled"
        );
    }

    // ModelOpt-exported checkpoints drop a sibling `hf_quant_config.json`
    // whose TOP LEVEL is already the quantization block.
    serve_phases::merge_sidecar_quant_config(&model_dir, &mut config);

    // GLM-5.3 has two deliberately disjoint physical sources. An explicit
    // GGUF directory always wins; otherwise the presence of both EXL3 marker
    // files selects the strict 2.05-bpw admission. A partial marker set is a
    // broken EXL3 checkpoint, not permission to fall through to GGUF.
    let glm53_exl3_files =
        if config.model_type == "glm5_next" && std::env::var_os("ATLAS_GLM53_GGUF_DIR").is_none() {
            let index = model_dir.join("model.safetensors.index.json");
            let quant = model_dir.join("quantization_config.json");
            match (index.is_file(), quant.is_file()) {
                (true, true) => Some(
                    spark_model::weight_loader::admit_glm53_exl3_files(&model_dir)
                        .context("admitting the exact GLM-5.3 EXL3 checkpoint")?,
                ),
                (false, false) => None,
                _ => anyhow::bail!(
                    "GLM-5.3 EXL3 checkpoint is incomplete: both {} and {} are required",
                    index.display(),
                    quant.display(),
                ),
            }
        } else {
            None
        };
    let is_glm53_exl3 = glm53_exl3_files.is_some();
    // GLM-5.3 speculates only through its own DFlash2 drafter and policy
    // verifier (EXL3 checkpoint, `--dflash --dflash-gamma 8`); the generic
    // MTP/DFlash proposers never run against a GLM target.
    let glm_dflash_max_drafts = serve_phases::resolve_glm_dflash(
        &config.model_type,
        is_glm53_exl3,
        args.dflash,
        args.speculative,
        args.dflash_gamma,
        args.glm_dflash_max_drafts,
        serve_phases::GlmDflashRuntimeOptions {
            max_seq_len: args.max_seq_len,
            dspark_verify_mode,
            quantization: &args.dflash_quantization,
            window_size: args.dflash_window_size,
            method: &args.dflash_method,
            ddtree_budget: args.ddtree_budget,
            has_donor_model: args.dflash_donor_model.is_some(),
            mtp_gate: args.mtp_gate.as_deref(),
        },
    )?;

    let context_extension = super::context_extension::apply_context_extension(
        &mut config,
        args.max_seq_len,
        args.rope_theta_override,
        args.rope_yarn_factor,
    )?;
    let context_admission = super::context_extension::validate_context_extension_runtime(
        context_extension,
        super::context_extension::ContextRuntimeMode {
            speculative: args.speculative,
            dflash: args.dflash,
            self_speculative: args.self_speculative,
            ngram_speculative: args.ngram_speculative,
            high_speed_swap: args.high_speed_swap,
            hss_cache_blocks_per_seq: args.high_speed_swap_cache_blocks_per_seq,
            block_size: args.block_size,
            max_batch_size: args.max_batch_size,
            max_seq_len: args.max_seq_len,
            config_capacity: config.max_position_embeddings,
        },
    )?;
    if let Some(extension) = context_extension {
        match extension {
            super::context_extension::ContextExtension::Yarn {
                native_tokens,
                requested_tokens,
                factor,
                original_tokens,
                attention_factor,
            } => tracing::warn!(
                "static YaRN context extension: native={} requested={} factor={} \
                 original={} attention_factor={}; capacity enabled, retrieval qualification required",
                native_tokens,
                requested_tokens,
                factor,
                original_tokens,
                attention_factor,
            ),
            super::context_extension::ContextExtension::Theta {
                native_tokens,
                requested_tokens,
                rope_theta,
            } => tracing::warn!(
                "EXPERIMENTAL theta context extension: native={} requested={} rope_theta={}; \
                 capacity is enabled but long-context quality is not qualified",
                native_tokens,
                requested_tokens,
                rope_theta,
            ),
        }
    }

    if let Some(ref qc) = config.quantization_config {
        tracing::info!(
            "Quantization config: method={:?}, algo={:?}, format={:?}, {} module(s) in ignore list",
            qc.quant_method,
            qc.quant_algo,
            qc.format,
            qc.ignore_modules.len(),
        );
    }

    tracing::info!(
        "Model config: {} layers, {} attention, {} SSM, {} experts, rope_theta={}, head_dim={}, rotary_dim={}",
        config.num_hidden_layers,
        config.num_attention_layers(),
        config.num_ssm_layers(),
        config.num_experts,
        config.rope_theta,
        config.head_dim,
        config.rotary_dim(),
    );

    // 2. Select kernel target and initialize GPU backend
    spark_runtime::progress::phase(3, "gpu init");
    //
    // Default dispatch: each kernel target declares which (model_type,
    // hidden_size) pairs it supports via [[model_types]] in MODEL.toml.
    // Exact hidden_size matches win over wildcards.
    //
    // CLI override: `--kernel-target <NAME>` forces a specific target
    // dir, useful for abliterated/heretic checkpoints whose tuned
    // MODEL.toml defaults would otherwise be shadowed by the canonical
    // target with the same `(model_type, hidden_size)` tuple.
    let ptx_set = if let Some(ref forced) = args.kernel_target {
        // Exact name match first, then substring fallback. Exact-first
        // is important for abliterated targets whose names are
        // supersets of the canonical (e.g. `qwen3.6-35b-a3b-abl`
        // contains `qwen3.6-35b-a3b`).
        atlas_kernels::available_targets()
            .into_iter()
            .find(|t| t.target.model == *forced)
            .or_else(|| atlas_kernels::ptx_for_model(forced))
            .with_context(|| {
                format!(
                    "--kernel-target '{}' did not match any compiled kernel target. \
                     Available targets: {:?}",
                    forced,
                    atlas_kernels::available_targets()
                        .iter()
                        .map(|t| &t.target.model)
                        .collect::<Vec<_>>(),
                )
            })?
    } else {
        atlas_kernels::ptx_for_config(&config.model_type, config.hidden_size).with_context(
            || {
                format!(
                    "No compiled kernel target matches model_type '{}' / hidden_size={}. \
                     Available targets: {:?}",
                    config.model_type,
                    config.hidden_size,
                    atlas_kernels::available_targets()
                        .iter()
                        .map(|t| &t.target.model)
                        .collect::<Vec<_>>(),
                )
            },
        )?
    };
    // GLM-5.3 ships two kernel bundles (exl3, iq3) declaring the same
    // (model_type, hidden_size); a wildcard build contains both, so pick the
    // one matching the physical source rather than whichever sorts first.
    let ptx_set = if config.model_type == "glm5_next" {
        let required_quant = if is_glm53_exl3 { "exl3" } else { "iq3" };
        if ptx_set.target.quant == required_quant {
            ptx_set
        } else {
            atlas_kernels::available_targets()
                .into_iter()
                .find(|t| {
                    t.target.model == ptx_set.target.model && t.target.quant == required_quant
                })
                .with_context(|| {
                    format!(
                        "GLM-5.3 serving selected the {required_quant} physical source, but this \
                         binary has no {} {required_quant} kernel bundle. Rebuild with \
                         ATLAS_TARGET_QUANT={required_quant} (or '*') and restart.",
                        ptx_set.target.model,
                    )
                })?
        }
    } else {
        ptx_set
    };
    let sampling_presets = ptx_set.sampling;
    tracing::info!(
        "Selected kernel target: {} ({} modules)",
        ptx_set.target,
        ptx_set.modules.len(),
    );
    // Record the (model, quant) this serve path actually resolved, so the
    // TUI Kernels tab can look up its exact module set instead of falling
    // back to whichever target sorts first in a multi-target build.
    crate::tui::data::kernels::publish_loaded_target(ptx_set.target.model, ptx_set.target.quant);

    // Apply MODEL.toml [behavior].default_num_drafts unless user passed --num-drafts.
    serve_phases::apply_model_default_num_drafts(&mut args, &ptx_set);
    crate::recipe_contract::resolved::check_num_drafts(args.speculative, args.num_drafts)?;
    let tc_parity_ar_reference_rows =
        serve_phases::admit_tc_parity_ar_reference(&args, &config, ptx_set.target.model)?;

    // Flash-Next's four-stream/PLE execution has no qualified auxiliary
    // self-spec or inherited K=2 n-gram verifier. Diagnostic escape hatches
    // are inappropriate for a serving claim: fail before GPU/model loading.
    if qwen4_aux_speculation_unsupported(
        args.self_speculative,
        args.ngram_speculative,
        config.is_qwen4_exp(),
    ) {
        anyhow::bail!(
            "--self-speculative/--ngram-speculative are not losslessly qualified for qwen4_exp; \
             use native --speculative and/or --dflash"
        );
    }
    let explicit_speculation =
        args.speculative || args.dflash || args.self_speculative || args.ngram_speculative;
    let qwen4_loaded_serial_control = qwen4_native_loaded_serial_control_requested(
        args.speculative,
        args.dflash,
        args.self_speculative,
        args.ngram_speculative,
        config.is_qwen4_exp(),
    );
    // Resolve the process-global kill switch strictly exactly once before
    // model construction.  The same immutable value is passed through native
    // admission and into the scheduler, so non-Transformer internal
    // speculative models (notably DSV4.1/DSpark) retain the global control.
    let speculation_forced_disabled = serve_phases::speculation_forced_disabled()?;
    if explicit_speculation && speculation_forced_disabled && !qwen4_loaded_serial_control {
        anyhow::bail!(
            "an explicit speculative mode was requested, but ATLAS_DISABLE_SPECULATION=1 would \
             silently force serial decode; only the Qwen4 native-MTP loaded-serial control is \
             admitted, otherwise unset the kill switch or remove the speculative flag"
        );
    }

    // Every pool the factory allocates from here on is recorded for the
    // end-of-load budget table; a swap must not report the previous model.
    spark_runtime::startup_ledger::reset();
    let (gpu, free_mem) = serve_phases::init_gpu_backend(&args, &ptx_set)?;
    // Kept alongside the scheduler handle in `Prepared` so a swap can free
    // every allocation this specific backend instance made, once it is
    // provably quiescent — see `GpuBackend::free_all_allocations` and
    // `model_swap::swap`. An `Arc` clone, not the original: `gpu` below is
    // still moved into the model by value.
    let gpu_handle = gpu.clone();

    // Now that the GPU backend is initialized and device topology / compute
    // capability are known:
    // - Arm the in-process MemAvailable watcher (disabled by default on discrete
    //   GPUs unless `FLUX_MEM_WATCH=on`).
    // - Finalize FlashInfer SM121 artifact preflight with the active device's
    //   compute capability (fail-closed on `(aarch64, SM121)`, cleanly bypassed
    //   and disabled on `x86_64`, `SM120`, and `SM103`).
    super::mem_watch::start_for_gpu(gpu.as_ref())?;
    super::flashinfer_artifact::preflight_live_for_device(
        &config.model_type,
        gpu.compute_capability(),
    )
    .map_err(anyhow::Error::msg)?;

    // 2a. Resolve TP / EP / PP topology BEFORE preflight reserve & preload fit check
    // so per-rank sharded layers, experts, KV heads, and QSA side-cache sizes are
    // accurately reflected in `FIT_ESTIMATE` / `FIT_WARN` and `preflight_reserve`.
    spark_runtime::progress::phase(4, "topology");
    let serve_phases::Topology {
        world_size,
        tp_size: _tp_size,
        ep_size,
        tp_rank: _tp_rank,
        ep_rank,
    } = serve_phases::resolve_topology(&args, &mut config)?;
    // FP8 KV calibration: CLI flag overrides MODEL.toml default.
    config.fp8_kv_calibration_tokens = if args.fp8_kv_calibration_tokens > 0 {
        args.fp8_kv_calibration_tokens
    } else {
        ptx_set.behavior.fp8_kv_calibration_tokens
    };

    // ── Pre-load reserve preflight ──
    let serve_phases::ReservePreflight {
        inference_reserve,
        buffer_arena_bytes,
        gdn_two_phase_bytes,
        prefill_budget,
        max_batch_tokens,
        breakdown: reserve_breakdown,
    } = serve_phases::preflight_reserve(&args, &config, free_mem)?;
    crate::recipe_contract::resolved::check_prefill_budget(prefill_budget)?;
    let total_reserve = inference_reserve + buffer_arena_bytes;
    // Before the weight store allocates: refuse a load that cannot fit
    // above the MemAvailable floor (FLUX_FIT_CHECK: warn by default, refuse opt-in).
    if let Some(need) = serve_phases::run_preload_fit_check(
        &args,
        &config,
        &model_dir,
        &reserve_breakdown,
        buffer_arena_bytes,
        inference_reserve,
        gpu.as_ref(),
    )? && let Some(watch) = super::mem_watch::shared()
    {
        watch.set_fit_need(need);
    }

    // 2a-2. OOM watchdog: background async task that polls GPU memory every 2s.
    // On GB10 unified memory, GPU OOM = system freeze, so we exit(1) early.
    // Threshold: 2 GB (enough to detect runaway allocation before system locks up).
    //
    // Discrete GPUs have dedicated VRAM isolated from Linux host RAM and fail
    // cleanly with CUDA OOM instead of hanging the host; running a 2 GB kill
    // threshold would falsely terminate healthy 16–32 GB GPUs at 90% utilization.
    //
    // CUDA-only: Apple Silicon UMA already exposes `currentAllocatedSize`
    // and the OS handles memory pressure via Metal's working-set policy,
    // so the dedicated watchdog isn't needed.
    #[cfg(feature = "cuda")]
    let oom_watchdog = if gpu.is_unified_memory() {
        tracing::info!("OOM watchdog started (threshold: 2 GB, interval: 2s)");
        Some(OomWatchdogGuard(
            spark_runtime::cuda_backend::spawn_oom_watchdog(
                2048, // 2 GB threshold
                std::time::Duration::from_secs(2),
            ),
        ))
    } else {
        tracing::info!(
            "OOM watchdog skipped on discrete GPU (dedicated VRAM fails cleanly with CUDA OOM)"
        );
        None
    };
    #[cfg(not(feature = "cuda"))]
    let oom_watchdog: Option<OomWatchdogGuard> = None;

    // 3. Load model weights
    spark_runtime::progress::phase(5, "weight load");
    if config.is_qwen4_exp() {
        let manifest = model_dir.join("ple-offload/manifest.json");
        if manifest.is_file() {
            config.ple_offload_manifest = Some(manifest.to_string_lossy().into_owned());
            tracing::info!(
                "Qwen4 PLE: discovered sparse offload manifest {}",
                manifest.display()
            );
        }
    }
    let effective_oom_guard_mb = effective_discrete_oom_guard_mb(
        args.oom_guard_mb,
        gpu.total_memory().ok(),
        !gpu.is_unified_memory(),
    );
    let oom_reserve_bytes = effective_oom_guard_mb * 1024 * 1024;
    if effective_oom_guard_mb != args.oom_guard_mb {
        tracing::info!(
            "OOM guard reserve capped on discrete GPU: {} MB (configured: {} MB)",
            effective_oom_guard_mb,
            args.oom_guard_mb,
        );
    } else {
        tracing::info!("OOM guard reserve: {} MB", effective_oom_guard_mb);
    }
    // GLM-5.3-Flash's trunk (34 KDA + 11 DSA layers under mHC) cannot use the
    // generic transformer path. It loads through its own EXL3 or GGUF device
    // store and is built by `build_glm53_*` below. Every other model takes the
    // unchanged safetensors path.
    let glm53_exl3 = if let Some(files) = glm53_exl3_files {
        let physical_vocab = spark_runtime::weights::gguf::GLM53_GGUF_VOCAB_SIZE;
        if config.vocab_size != physical_vocab {
            tracing::info!(
                "GLM-5.3 EXL3: config.json declares vocab_size={}, physical embedding holds {} — using the payload",
                config.vocab_size,
                physical_vocab,
            );
            config.vocab_size = physical_vocab;
        }
        let loaded = match spark_model::weight_loader::load_glm53_exl3_store(
            &files,
            gpu.as_ref(),
            oom_reserve_bytes,
        ) {
            Ok(store) => store,
            Err(error) => match error.retry_cleanup(gpu.as_ref()) {
                Ok(primary) => return Err(primary.context("loading GLM-5.3 EXL3 checkpoint")),
                Err(retained) => anyhow::bail!("GLM-5.3 EXL3 load cleanup failed: {retained}"),
            },
        };
        tracing::info!(
            "GLM-5.3 EXL3 store: {} tensors, {} bytes resident",
            loaded.len(),
            loaded.allocated_bytes(),
        );
        Some((files, loaded))
    } else {
        None
    };
    let glm53_gguf: Option<(
        spark_runtime::weights::gguf::Glm53QuantProfile,
        spark_runtime::weights::gguf::GgufDeviceStore,
    )> = if config.model_type == "glm5_next" && glm53_exl3.is_none() {
        let profile = spark_runtime::weights::gguf::Glm53QuantProfile::UdIq2Xxs;
        // The GGUF payload does not have to sit beside config.json: search an
        // explicit ATLAS_GLM53_GGUF_DIR, the profile subdirectory, then the
        // model directory itself.
        let shard_dir = std::env::var_os("ATLAS_GLM53_GGUF_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| model_dir.join(profile.directory_name()));
        let shard_dir = if shard_dir.is_dir() {
            shard_dir
        } else {
            model_dir.clone()
        };
        tracing::info!("GLM-5.3 GGUF shard directory: {}", shard_dir.display());
        let gguf_vocab = spark_runtime::weights::gguf::GLM53_GGUF_VOCAB_SIZE;
        if config.vocab_size != gguf_vocab {
            tracing::info!(
                "GLM-5.3: config.json declares vocab_size={}, GGUF payload holds {} — \
                 using the payload",
                config.vocab_size,
                gguf_vocab,
            );
            config.vocab_size = gguf_vocab;
        }
        let shards = super::glm53_gguf_resolver::resolve_exact_glm53_shards(&shard_dir, profile)
            .context("resolving GLM-5.3 GGUF shards")?;
        let loaded = serve_phases::TargetStoreLoadPlan::resolved_glm53(profile, shards)
            .context("planning the GLM-5.3 GGUF load")?
            .load_glm53_gguf(gpu.as_ref(), oom_reserve_bytes)
            .map_err(|error| anyhow::anyhow!("GLM-5.3 GGUF load failed: {error}"))?;
        tracing::info!(
            "GLM-5.3 GGUF store: {} tensors, {} bytes resident",
            loaded.tensor_count(),
            loaded.total_bytes(),
        );
        Some(loaded.into_glm53_gguf()?)
    } else {
        None
    };
    let has_glm53_target_store = glm53_exl3.is_some() || glm53_gguf.is_some();
    let mut store = if has_glm53_target_store {
        // The safetensors-shaped checks below are skipped for GLM; this empty
        // store keeps their types satisfied without pretending to hold weights.
        spark_runtime::weights::WeightStore::empty()
    } else {
        spark_runtime::startup_ledger::measure(
            "weights/store",
            "whole safetensors store; per-family logical bytes are the weights/* rows",
            None,
            || {
                serve_phases::load_weight_store(
                    &args,
                    &config,
                    &model_dir,
                    gpu.as_ref(),
                    ep_rank,
                    ep_size,
                    oom_reserve_bytes,
                )
            },
        )?
    };

    let has_in_store_mtp_experts = store.contains("mtp.layers.0.mlp.experts.0.gate_proj.weight")
        || store.contains("mtp.layers.0.mlp.experts.gate_up_proj");
    let mtp_source = super::mtp_source::MtpSource::from_env()?;
    let wants_mtp_sidecar = mtp_source.wants_sidecar(has_in_store_mtp_experts);
    let resolved_mtp_dir: Option<std::path::PathBuf> = if config.pp_rank > 0 {
        None
    } else if let Some(explicit) = &args.mtp_from_path {
        if explicit.is_dir() {
            Some(explicit.clone())
        } else if has_in_store_mtp_experts && mtp_source != super::mtp_source::MtpSource::Sidecar {
            tracing::info!(
                requested = %explicit.display(),
                "MTP routed experts already present in target checkpoint; skipping missing --mtp-from-path"
            );
            None
        } else {
            let fallback_candidates = [
                model_dir
                    .parent()
                    .map(|p| p.join("Qwen3.8-Flash-Next-Inferact-MTP")),
                Some(std::path::PathBuf::from(
                    "/models/Qwen3.8-Flash-Next-Inferact-MTP",
                )),
                Some(std::path::PathBuf::from(
                    "/opt/flux/models/Qwen3.8-Flash-Next-Inferact-MTP",
                )),
            ];
            let resolved = fallback_candidates
                .into_iter()
                .flatten()
                .find(|cand| cand.is_dir());
            if let Some(ref found) = resolved {
                tracing::info!(
                    requested = %explicit.display(),
                    resolved = %found.display(),
                    "Resolved missing --mtp-from-path to bundled/companion MTP sidecar"
                );
            }
            resolved.or_else(|| Some(explicit.clone()))
        }
    } else if args.speculative
        && !args.dflash
        && !args.self_speculative
        && !args.ngram_speculative
        && config.is_qwen4_exp()
        && wants_mtp_sidecar
    {
        let auto_candidates = [
            model_dir
                .parent()
                .map(|p| p.join("Qwen3.8-Flash-Next-Inferact-MTP")),
            Some(std::path::PathBuf::from(
                "/models/Qwen3.8-Flash-Next-Inferact-MTP",
            )),
            Some(std::path::PathBuf::from(
                "/opt/flux/models/Qwen3.8-Flash-Next-Inferact-MTP",
            )),
        ];
        let found = auto_candidates
            .into_iter()
            .flatten()
            .find(|cand| cand.is_dir());
        if let Some(ref dir) = found {
            tracing::info!(
                resolved = %dir.display(),
                "Auto-resolved Qwen4 native MTP expert sidecar for checkpoint without in-store MTP experts"
            );
        }
        found
    } else {
        None
    };

    let mtp_sidecar_plan = match &resolved_mtp_dir {
        Some(mtp_dir) => {
            anyhow::ensure!(args.speculative, "--mtp-from-path requires --speculative");
            anyhow::ensure!(
                config.is_qwen4_exp(),
                "--mtp-from-path currently supports only qwen4_exp"
            );
            let plan = super::mtp_source::plan_sidecar(mtp_source, has_in_store_mtp_experts)?;
            if plan == super::mtp_source::MtpSidecarPlan::Skip {
                tracing::warn!(
                    "QWEN4_MTP_SOURCE source=checkpoint policy={} sidecar={} \
                     reason=target_has_expert_bank; set FLUX_MTP_SOURCE=sidecar to replace it",
                    mtp_source.as_str(),
                    mtp_dir.display(),
                );
            }
            plan
        }
        None => super::mtp_source::MtpSidecarPlan::Skip,
    };
    if let Some(mtp_dir) = resolved_mtp_dir
        .as_ref()
        .filter(|_| mtp_sidecar_plan != super::mtp_source::MtpSidecarPlan::Skip)
    {
        use spark_runtime::weights::WeightLoader;
        let expert_prefixes =
            (0..config.num_experts).map(|expert| format!("mtp.layers.0.mlp.experts.{expert}."));
        let mut loader = spark_runtime::fast_weights::FastSafetensorsLoader::new()
            .with_name_prefixes(expert_prefixes);
        loader.peak_memory_multiplier = Some(1.15);
        let mtp_store = spark_runtime::startup_ledger::measure(
            "weights/mtp-sidecar",
            format!("--mtp-from-path {}", mtp_dir.display()),
            None,
            || {
                loader
                    .load(mtp_dir, gpu.as_ref(), oom_reserve_bytes)
                    .with_context(|| format!("Failed to load MTP sidecar {}", mtp_dir.display()))
            },
        )?;
        let layout = spark_model::weight_loader::qwen4_mtp::validate_qwen4_mtp_expert_store(
            &mtp_store, &config,
        )
        .context("Qwen4 MTP expert sidecar schema validation failed")?;
        let count = mtp_store.len();
        let bytes = mtp_store.total_bytes();
        let mut replaced = 0usize;
        let mut freed_bytes = 0usize;
        if mtp_sidecar_plan == super::mtp_source::MtpSidecarPlan::Replace {
            // The sidecar is loaded and schema-valid before the target bank is
            // touched, so a bad sidecar cannot leave a half-emptied store.
            for (name, tensor) in store.take_prefix(super::mtp_source::MTP_EXPERT_PREFIX)? {
                freed_bytes += tensor.byte_size();
                gpu.free(tensor.ptr)
                    .with_context(|| format!("free replaced checkpoint MTP tensor '{name}'"))?;
                replaced += 1;
            }
        }
        store.merge_disjoint(mtp_store)?;
        spark_model::weight_loader::qwen4_mtp::validate_qwen4_mtp_store(&store, &config)
            .context("Merged Qwen4 MTP schema validation failed")?;
        tracing::info!(
            "QWEN4_MTP_SOURCE source=sidecar policy={} plan={:?} tensors={count} \
             replaced={replaced} freed_gib={:.3} path={}",
            mtp_source.as_str(),
            mtp_sidecar_plan,
            freed_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            mtp_dir.display(),
        );
        tracing::info!(
            tensors = count,
            gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            layout = ?layout,
            path = %mtp_dir.display(),
            "Qwen4 native MTP sidecar loaded and schema-validated"
        );
    } else if args.speculative && config.is_qwen4_exp() && has_in_store_mtp_experts {
        tracing::info!(
            "QWEN4_MTP_SOURCE source=checkpoint policy={}",
            mtp_source.as_str()
        );
    }

    // 3b. Auto-detect weight key prefix for nested models. Safetensors only:
    // the GLM store has its own pinned tensor names.
    if !has_glm53_target_store {
        serve_phases::auto_detect_weight_prefix(&store, &mut config);
    }

    // Pre-flight weight-store / config consistency check. Runs before
    // NCCL init so a mis-matched checkpoint (wrong expert count, MiniMax
    // + MTP tensors + `--speculative`, missing embedding, etc.) aborts
    // this rank with a readable error BEFORE rank 1 ever connects or
    // `ncclCommInitRank` is called. Several community re-quants of
    // MiniMax M2.7 hang on NCCL init today because the actual mismatch
    // only surfaces later inside `build_model`; this check surfaces it
    // up-front.
    if !has_glm53_target_store {
        spark_model::preflight::preflight(&store, &config, args.speculative)
            .context("Checkpoint pre-flight check failed")?;
    }

    // Resolve and log the QuantFormat dispatch decision now so a silent
    // fallback is visible in the server log (and not just in the
    // detection code path mid-load). The returned trait object is
    // currently only consulted via `detect_nvfp4_variant`; explicit
    // use at each load site is a follow-up migration.
    if has_glm53_target_store {
        // GLM's weights live in its own EXL3/GGUF store; `store` is the empty
        // placeholder, so the safetensors heuristic would warn about an
        // unrecognized quant_method and a runtime BF16->NVFP4 fallback that
        // never happens.
        tracing::info!(
            "Quantization format: {} (GLM-5.3 target store, weights served as stored; no runtime NVFP4 quantization)",
            if glm53_exl3.is_some() { "exl3" } else { "gguf" },
        );
    } else {
        let quant_format = spark_model::quant_format::detect_quant_format(&config, &store);
        tracing::info!(
            "Quantization format: {} (base variant {:?}), ignored globs = {}",
            quant_format.name(),
            quant_format.base_variant(),
            match &config.quantization_config {
                Some(qc) => qc.ignore_modules.len(),
                None => 0,
            },
        );
    }

    // 4. Post-load OOM check + audit log.
    let device_store_bytes = match (&glm53_exl3, &glm53_gguf) {
        (Some((_, exl3_store)), _) => exl3_store.allocated_bytes(),
        (None, Some((_, gguf_store))) => gguf_store.total_bytes(),
        (None, None) => store.total_bytes(),
    };
    serve_phases::post_load_memory_audit(
        &args,
        &config,
        gpu.as_ref(),
        device_store_bytes,
        free_mem,
        inference_reserve,
        total_reserve,
        gdn_two_phase_bytes,
        max_batch_tokens,
        &reserve_breakdown,
    )?;

    // 5. Build model via factory.
    spark_runtime::progress::phase(6, "kv cache");
    serve_phases::validate_qwen4_prefill_geometry(&config, prefill_budget)?;
    // `--mtp-gate` accepts a closed set; reject unknown values early rather
    // than silently treating them as "no gate".
    const MTP_GATES: &[&str] = &["auto", "force", "dflash", "mtp"];
    if let Some(gate) = args.mtp_gate.as_deref()
        && !MTP_GATES.contains(&gate)
    {
        anyhow::bail!(
            "--mtp-gate {gate}: unknown value (expected one of {MTP_GATES:?}). \
             Omit the flag to run with no gate."
        );
    }
    if args.mtp_gate.is_some() && !(args.speculative || args.dflash) {
        anyhow::bail!(
            "--mtp-gate needs speculation: pass --dflash and/or --speculative, \
             or omit --mtp-gate."
        );
    }
    if args.dflash && args.enable_prefix_caching {
        anyhow::bail!(
            "--dflash cannot be combined with --enable-prefix-caching: the combination has a \
             known SM12.x correctness regression on multi-turn cache hits"
        );
    }
    let prefix_cache = serve_phases::build_prefix_cache(&args);
    let comm = serve_phases::init_nccl_comm(&args, gpu.as_ref(), world_size)?;
    // Not for GLM-5.3: the payload pads the embedding to 154,880 physical rows
    // while tokenizer.json holds 154,856 tokens; capping here would undo the
    // physical-vocab override above and refuse a correct checkpoint.
    if !has_glm53_target_store {
        serve_phases::cap_vocab_size_to_tokenizer(&model_dir, &mut config);
    }
    let serve_phases::KvCacheConfig {
        effective_kv_dtype_str,
        kv_dtype,
        layer_dtypes,
        hss_cache_blocks_per_seq: _,
    } = serve_phases::resolve_kv_cache_config(&args, &config, ptx_set.behavior.default_kv_dtype)?;
    crate::recipe_contract::resolved::check_kv(&effective_kv_dtype_str)?;
    // GLM-5.3's DFlash2 drafter is owned by the GLM model (installed by its
    // factory below), not by the generic DFlash builder.
    let glm53_dflash2_dir = if is_glm53_exl3 && args.dflash {
        Some(serve_phases::resolve_dflash_drafter_dir(&args, &ptx_set)?)
    } else {
        None
    };
    let dflash_drafter_state = if glm53_dflash2_dir.is_some() {
        None
    } else {
        serve_phases::load_dflash_drafter(
            &args,
            &ptx_set,
            gpu.as_ref(),
            dspark_verify_mode,
            tc_parity_ar_reference_rows,
        )?
    };
    let dflash_donor_state = serve_phases::load_dflash_donor(&args, gpu.as_ref())?;
    if let Some((_, drafter_config)) = dflash_drafter_state.as_ref() {
        args.dflash_gamma = Some(drafter_config.resolve_draft_count(args.dflash_gamma)?);
    }
    if config.is_qwen4_exp()
        && dflash_donor_state.is_some()
        && !matches!(args.dflash_gamma, Some(1 | 2))
    {
        anyhow::bail!(
            "the experimental Qwen4-to-dense-V3 bridge supports only --dflash-gamma 1 or 2 \
             (physical verifier K=2 or K=3); got {:?}",
            args.dflash_gamma
        );
    }
    let dflash_quant_raw = std::env::var("ATLAS_DFLASH_QUANTIZATION")
        .ok()
        .or_else(|| std::env::var("FLUX_DFLASH_QUANTIZATION").ok())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| args.dflash_quantization.clone());
    let dflash_quantization = match dflash_quant_raw.trim().to_ascii_lowercase().as_str() {
        "bf16" => spark_model::layers::DflashQuantization::Bf16,
        "fp8" => spark_model::layers::DflashQuantization::Fp8,
        "nvfp4" => spark_model::layers::DflashQuantization::Nvfp4,
        other => anyhow::bail!(
            "unknown DFlash quantization value `{other}`; expected `bf16`, `fp8`, or `nvfp4`"
        ),
    };
    let dflash_draft_sample_method = args
        .dflash_draft_sample_method
        .parse::<spark_model::dflash_sampling::DflashDraftSampleMethod>()
        .map_err(anyhow::Error::msg)?;
    let dflash_args =
        dflash_drafter_state
            .as_ref()
            .map(|(s, c)| spark_model::factory::DflashBuildArgs {
                drafter_store: s,
                donor_store: dflash_donor_state.as_ref(),
                drafter_config: c.clone(),
                gamma: args.dflash_gamma,
                dspark_verify_mode,
                window_size: if args.dflash_window_size > 0 {
                    Some(args.dflash_window_size)
                } else {
                    None
                },
                quantization: dflash_quantization,
                draft_sample_method: dflash_draft_sample_method,
                probabilistic_qualification_enabled: args.experimental_dflash_probabilistic,
            });
    // GLM-5.3 bypasses the generic transformer builder: `Glm53Model`
    // implements `Model` directly over the owned device store loaded above.
    let model = if let Some((files, exl3_store)) = glm53_exl3 {
        spark_model::factory::build_glm53_exl3_model(
            files,
            exl3_store,
            gpu,
            args.max_seq_len,
            glm53_dflash2_dir.as_deref(),
        )
        .context("building the GLM-5.3 EXL3 model")?
    } else if let Some((profile, gguf_store)) = glm53_gguf {
        spark_model::factory::build_glm53_model(profile, &config, gguf_store, gpu, args.max_seq_len)
            .context("building the GLM-5.3 model")?
    } else {
        serve_phases::build_model(
            &args,
            &config,
            context_admission,
            &store,
            gpu,
            max_batch_tokens,
            kv_dtype,
            inference_reserve,
            layer_dtypes,
            prefix_cache,
            comm,
            dflash_args,
        )?
    };

    // Explicit native-MTP and DFlash requests are qualification
    // contracts, not best-effort hints. Refuse to start if construction left
    // the model plain, installed the wrong proposer kind, or lost a dual arm.
    // Target-only prefill never enters this gate.
    serve_phases::validate_speculation_admission(
        &args,
        &config,
        model.as_ref(),
        speculation_forced_disabled,
    )?;
    // The one startup budget table, after every factory pool exists; fails
    // closed under the UMA MemAvailable floor (GB10) or discrete VRAM floor.
    serve_phases::audit_startup_budget(
        &config,
        &store,
        device_store_bytes,
        &reserve_breakdown,
        gpu_handle.as_ref(),
    )?;
    if args.self_speculative && !model.has_self_speculative() {
        anyhow::bail!(
            "--self-speculative was requested, but this model has no supported layer-skipping path; \
             pure-attention and qwen4_exp models fail closed"
        );
    }
    if args.ngram_speculative && !model.supports_ngram_speculative() {
        anyhow::bail!(
            "--ngram-speculative was requested, but this backend has no qualified K=2 \
             verify/checkpoint contract"
        );
    }

    // `--check-kernels`: every kernel lookup is eager and lives in a layer
    // constructor on the `build_model` path above, so the audit is complete
    // exactly here — and a check that ran any earlier would resolve a
    // DIFFERENT set than a real serve. Prints the report and exits with the
    // unresolved count; never returns. The scheduler is not started and no
    // port is bound.
    if args.check_kernels {
        serve_phases::check_and_exit(&ptx_set);
    }
    spark_runtime::progress::phase(7, "kernel audit");

    // Build HSS config early so the EP worker can install it.
    let early_high_speed_swap_cfg = serve_phases::build_high_speed_swap_config(&args)?;

    // EP worker: rank > 0 enters command loop, returns when head exits.
    let mut model_opt = Some(model);
    if serve_phases::maybe_run_ep_worker(&args, &mut model_opt, &early_high_speed_swap_cfg)? {
        // An EP worker (rank > 0) never serves HTTP: it ran its command loop and
        // the head has exited. `None` = nothing for the async tail to do.
        return Ok(None);
    }
    let model = model_opt.expect("head retains model on rank 0");
    if config.is_qwen4_exp() {
        super::qwen4_concurrency::log_qwen4_batch_plan(&*model, args.max_batch_size)?;
    }

    // TQ+ InnerQ: opt-in via `TURBO_INNERQ=N` (N = calibration token count).
    // Once enabled, the kernel-side apply pass starts accumulating K² stats
    // and the scheduler polls `maybe_finalize` per prefill chunk; once N
    // tokens have flowed through, scales activate and stay live for the
    // process lifetime. CUDA-only: the driver talks to the CUDA Driver API
    // directly via `atlas_core::registry`, which doesn't exist on metal.
    #[cfg(feature = "cuda")]
    if let Some(driver) = spark_model::layers::qwen3_attention::InnerQDriver::from_env() {
        match driver.start() {
            Ok(()) => {
                let _ = spark_model::layers::qwen3_attention::INNERQ.set(driver);
            }
            Err(e) => {
                tracing::warn!("InnerQ calibration disabled: start() failed: {e:#}");
            }
        }
    }

    // Build EOS token list from generation_config.json (authoritative) or config.json fallback
    let mut eos_tokens = serve_phases::load_eos_tokens(&model_dir, &config);

    // Read default sampling parameters from generation_config.json.
    let serve_phases::SamplingDefaults {
        temperature: default_temperature,
        top_k: default_top_k,
        top_p: default_top_p,
        top_n_sigma: default_top_n_sigma,
        min_p: default_min_p,
    } = serve_phases::load_sampling_defaults(&model_dir, &args, &sampling_presets.non_thinking);
    serve_phases::log_sampling_presets(&sampling_presets, default_min_p, default_top_n_sigma);

    // 6. Load tokenizer
    spark_runtime::progress::phase(8, "tokenizer");
    // Thinking support is derived from model capabilities, not hardcoded model names.
    // Models with SSM layers or Qwen3.5-style architecture support <think> tokens.
    // The --enable-thinking flag controls OPEN-ENDED vs CLOSED thinking.
    let caps = config.capabilities();
    let supports_thinking = caps.supports_thinking;
    let tokenizer = ChatTokenizer::from_model_dir(
        &model_dir,
        eos_tokens[0],
        supports_thinking,
        &config.model_type,
        Some(std::path::Path::new(".")), // repo root for override templates
    )?
    .with_special_tokens_in_content(
        super::content_specials::resolve_and_log(args.allow_special_tokens_in_content)?
            .allows_structural(),
    );

    // REST retrieval draft store (ATLAS_REST_STORE). Validated against the
    // tokenizer just loaded; a fingerprint mismatch aborts startup.
    serve_phases::init_rest_store(&model_dir)?;

    // Tokenizer-derived runtime: vocab cap, reasoning parser, think tokens,
    // im_start hard-stop, reflection suppression, tool-call open/close tokens,
    // and the XGrammar engine.
    let serve_phases::TokenizerRuntime {
        reasoning_parser_box,
        think_end_token,
        think_start_token,
        code_fence_token,
        reflection_suppress_ids,
        tool_call_start_token,
        tool_call_end_token,
        grammar_engine,
    } = serve_phases::resolve_tokenizer_runtime(
        &args,
        &ptx_set,
        &mut config,
        &tokenizer,
        &mut eos_tokens,
        supports_thinking,
    )?;

    // 7. Create scheduler channel + spawn scheduler
    spark_runtime::progress::phase(9, "scheduler");
    crate::scheduler::queue_gate::QUEUE_GATE.configure(args.max_queue_depth);
    crate::scheduler::kv_headroom::configure(args.kv_pool_tokens);
    if let Some(depth) = args.max_queue_depth {
        tracing::info!(
            "Request queue: up to {} running (--max-batch-size), {depth} waiting \
             (--max-queue-depth), then 429",
            args.max_batch_size
        );
    }
    let (request_tx, request_rx) = mpsc::channel::<InferenceRequest>(args.max_num_seqs);

    let model_name = serve_phases::resolve_model_name(&args, &config_json, &model_dir);

    let scheduler_model = model;
    let scheduler_eos = eos_tokens;
    // Allow continuous batching across both single-GPU and multi-GPU TP/EP
    // topologies (workers maintain lockstep multi-slot state via
    // `worker_extra_slots` and lockstep `decode_batch`).
    let max_batch_size = args.max_batch_size;
    anyhow::ensure!(
        max_batch_size <= spark_runtime::buffers::DECODE_META_MAX_ROWS,
        "--max-batch-size {max_batch_size} exceeds the qualified decode-metadata ceiling of {}",
        spark_runtime::buffers::DECODE_META_MAX_ROWS
    );
    // `use_speculative` gates the scheduler's `step_mtp` path which already
    // dispatches both MTP and DFlash proposers via the shared `DraftProposer`
    // trait + the `drafts.len() ≥ 4` ladder route to `step_verify_dflash`
    // (`scheduler::mtp_step::step_mtp`). So `--dflash` enables `use_speculative` too.
    let use_speculative = (args.speculative || args.dflash) && scheduler_model.has_proposer();
    // The scheduler's own route decision (scheduler::speculation_routes), taken while the
    // model is still here, for the /flux/modes serving block; it cannot drift from the
    // scheduler because it is the same function on the same inputs.
    let serving_routes = crate::scheduler::speculation_routes(
        speculation_forced_disabled,
        scheduler_model.has_internal_spec(),
        args.speculative || args.dflash,
        scheduler_model.has_proposer(),
    );
    let use_self_spec = args.self_speculative && scheduler_model.has_self_speculative();
    let use_ngram_spec = args.ngram_speculative;
    // `DraftProposer::propose` defines `num_drafts` as the actual maximum
    // number of speculative tokens, not verify width K. DFlash currently owns
    // its configured block width internally, but keep the scheduler value in
    // the same unit for early-exit and future proposer implementations.
    let num_drafts = if let Some(limit) = glm_dflash_max_drafts {
        tracing::info!(
            "GLM DFlash2 verify width: {limit} drafts; the checkpoint still proposes a block of 8"
        );
        limit
    } else if args.dflash {
        args.dflash_gamma.unwrap_or(15).max(1)
    } else {
        args.num_drafts
    };

    if args.dflash && !scheduler_model.has_proposer() {
        anyhow::bail!(
            "--dflash was requested, but the built model has no installed external proposer"
        );
    }
    if let Some(limit) = glm_dflash_max_drafts {
        tracing::info!(
            "GLM DFlash2 speculative decoding: ENABLED (block=8, verify_drafts={limit}, \
             window=2048 checkpoint-pinned, context_capture_capacity={} tokens, \
             max_sequence={} tokens, drafter installed)",
            spark_model::model::glm53::GLM53_DFLASH2_CONTEXT_CAPACITY,
            spark_model::model::glm53::GLM53_DFLASH2_MAX_SEQUENCE_TOKENS,
        );
    } else if args.dflash {
        tracing::info!(
            "DFlash speculative decoding: ENABLED (γ={}, window={}, drafter installed)",
            args.dflash_gamma.unwrap_or(15),
            if args.dflash_window_size == 0 {
                "full".to_string()
            } else {
                args.dflash_window_size.to_string()
            }
        );
    } else if use_ngram_spec {
        tracing::info!("N-gram speculative decoding: ENABLED (K=2 verify, CPU proposer)");
    } else if use_self_spec {
        tracing::info!(
            "Self-speculative decoding: ENABLED ({num_drafts} drafts/step, layer-skipping)"
        );
    } else if use_speculative {
        tracing::info!("Speculative decoding: ENABLED ({num_drafts} drafts/step)");
        // Loud warnings for known-degenerate gate combinations so users
        // don't enable them by accident expecting concurrency wins.
        if spark_model::layers::mtp_k3_batch_cseq_enabled() {
            tracing::warn!(
                "ATLAS_MTP_K3_BATCH_CSEQ=1 is set. The c-batched K=3 verify \
                 path is documented as ~600-1300 ms per K-step at c=2-3 \
                 (vs ~85 ms single-seq graphed). Without the multi-seq \
                 CUDA-graph unblocker (also see ATLAS_SSM_MULTI_SEQ_GRAPH), \
                 enabling this dramatically REDUCES throughput at \
                 concurrency. Recommend leaving unset until the indirection \
                 table fix lands upstream."
            );
        }
        if spark_model::layers::mtp_k2_batch_cseq_enabled() {
            tracing::warn!(
                "ATLAS_MTP_K2_BATCH_CSEQ=1 is set (K=2 sibling of the \
                 c-batched K=3 verify path). Same caveat applies — \
                 reduces throughput at concurrency until graph capture \
                 lands. Leave unset for production."
            );
        }
    } else if scheduler_model.has_proposer() {
        tracing::info!(
            "MTP proposer available but speculative decoding disabled (use --speculative to enable)"
        );
    }

    // ── Multi-seq SSM gate status notes ──
    // ATLAS_SSM_MULTI_SEQ_BATCHED is still un-wired (batched-projections
    // path scoped but not landed — would replace the per-seq QKVZ/BA/
    // out_proj launches with M=n batched GEMVs). Setting it is a no-op.
    // ATLAS_SSM_MULTI_SEQ_KERNEL is now LIVE: routes conv1d_update_l2norm
    // + gdn_decode through the FP32 multi-seq variants, collapsing those
    // two per-seq launches into one each at num_seqs ≥ 2. Wired in
    // layers/qwen3_ssm/trait_decode_multi_seq.rs.
    if spark_model::layers::ssm_multi_seq_batched_enabled() {
        tracing::warn!(
            "ATLAS_SSM_MULTI_SEQ_BATCHED=1 is set, but the batched-projections \
             multi-seq SSM decode path is not yet implemented — the per-seq \
             projection loop runs regardless, so setting this env var is \
             currently a no-op."
        );
    }
    if spark_model::layers::ssm_multi_seq_kernel_enabled() {
        tracing::info!(
            "ATLAS_SSM_MULTI_SEQ_KERNEL=1 is set. The multi-seq SSM \
             state-advance kernels (conv1d_update_l2norm_f32_multi_seq, \
             gdn_decode_f32_multi_seq) will be dispatched at num_seqs ≥ 2 \
             during multi-seq decode, collapsing 2 per-seq launches per \
             SSM layer per token to 1 each. Saves ~96 launches per token \
             at num_seqs=4 × 48 SSM layers."
        );
    }

    let policy: Box<dyn scheduling_policy::SchedulingPolicy> = match args.scheduling_policy.as_str()
    {
        "fifo" => {
            tracing::info!("Scheduling policy: FIFO");
            Box::new(scheduling_policy::FifoPolicy)
        }
        "slai" => {
            tracing::info!(
                "Scheduling policy: SLAI (TBT deadline={}ms)",
                args.tbt_deadline_ms,
            );
            Box::new(scheduling_policy::SlaiPolicy::new(args.tbt_deadline_ms))
        }
        other => anyhow::bail!(
            "Unknown scheduling policy '{}'. Supported: fifo, slai",
            other,
        ),
    };

    // Use prefill_budget (which accounts for SSM no-chunking override) instead of raw CLI arg.
    let max_prefill_tokens = prefill_budget;
    // Strict at load: a malformed cap is a configuration error, and a cap that
    // cannot shorten any chunk resolves to `None` so the scheduler log says so.
    let prefill_chunk_with_active_decode = scheduler::prefill_chunk_policy::resolve_chunk_cap(
        max_prefill_tokens,
        args.prefill_chunk_with_active_decode,
    )?;
    if args.prefill_chunk_with_active_decode != 0 && prefill_chunk_with_active_decode.is_none() {
        tracing::info!(
            "--prefill-chunk-with-active-decode {} cannot shorten a {}-token chunk; cap disabled",
            args.prefill_chunk_with_active_decode,
            max_prefill_tokens,
        );
    }
    let swap_space_gb = args.swap_space_gb;
    let block_size = args.block_size;

    // ── --high-speed-swap config validation ──
    let high_speed_swap_cfg = serve_phases::validate_head_high_speed_swap(
        &args,
        &early_high_speed_swap_cfg,
        swap_space_gb,
    )?;

    let adaptive_sampling = args.adaptive_sampling;
    let session_manager = session_manager::SessionSsmManager::new(600); // 10 min TTL
    // Spontaneous-thinking budget: when the model emits `<think>` without
    // the request having explicitly enabled thinking, this caps how many
    // thinking tokens are allowed before `</think>` is force-emitted. CLI
    // override beats MODEL.toml. Used by the scheduler in place of a
    // previous hard-coded 512 fallback so MODEL.toml can right-size the
    // cap per architecture.
    let scheduler_spontaneous_think_budget = args
        .max_thinking_budget
        .unwrap_or(ptx_set.behavior.max_thinking_budget);
    // Publish refreshable per-model behavior before the scheduler thread can
    // sample its first token. This executes on every load/swap; speculative
    // thinking also reads the value live rather than snapshotting it.
    serve_phases::publish_scheduler_behavior(&ptx_set.behavior);
    // Thinking nudge (FLUX_THINK_NUDGE, default off): installed or cleared
    // on every load, before the scheduler admits its first sequence.
    let think_nudge_setup = {
        let delimiters = [
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
        ];
        let structural: Vec<u32> = scheduler_eos
            .iter()
            .copied()
            .chain(delimiters.into_iter().flatten())
            .collect();
        serve_phases::configure_think_budget_nudge(
            &tokenizer,
            ptx_set.target.model,
            supports_thinking,
            &structural,
        )?;
        serve_phases::configure_think_nudge(
            &tokenizer,
            ptx_set.target.model,
            supports_thinking,
            &structural,
        )?
    };
    // Live-adjustable request defaults (/v1/flux/settings), seeded with the
    // startup values below; a configured FLUX_SETTINGS_FILE is reapplied here,
    // before the scheduler spawn, so a bad file fails the load cleanly.
    let app_behavior = {
        let mut b = ptx_set.behavior.clone();
        if let Some(cli_budget) = args.max_thinking_budget {
            b.max_thinking_budget = cli_budget;
        }
        b
    };
    let default_chat_template_kwargs = args
        .default_chat_template_kwargs
        .as_ref()
        .and_then(|s| crate::openai::ChatTemplateKwargs::from_json(s));
    let default_max_tokens = match args.default_max_tokens {
        Some(0) => None,
        Some(tokens) => Some(tokens),
        None => Some(args.max_seq_len.max(4_096)),
    };
    let runtime_settings = crate::runtime_settings::RuntimeSettings::new(
        crate::runtime_settings::StartupDefaults::new(
            &app_behavior,
            args.max_thinking_budget.is_some(),
            sampling_presets,
            default_chat_template_kwargs.as_ref(),
            args.disable_thinking,
            default_max_tokens,
            args.max_seq_len,
            if tokenizer.renders_qwen38_reasoning_effort() {
                crate::openai::qwen38_effort_thinking_budget
            } else {
                crate::openai::generic_effort_thinking_budget
            },
        ),
        crate::runtime_settings::NudgeControl::new(
            think_nudge_setup.tables,
            think_nudge_setup.startup_policy,
            think_nudge_setup.set_at_startup,
        ),
        crate::runtime_settings::SettingsFile::from_env()?,
        model_name.clone(),
    );
    runtime_settings.load_persisted()?;
    // Tool call parser resolution: CLI > MODEL.toml > defaults table.
    let tool_call_parser = serve_phases::resolve_tool_call_parser(&args, &ptx_set, &config)?;

    // deepseek_v41 image input: the preprocessing geometry from config.json's
    // vision_config. A checkpoint without one serves text only. Fallible, so
    // it runs before the scheduler spawn below.
    let dsv41 = config.model_type == "deepseek_v41";
    let dsv41_vision = if dsv41 {
        let raw: serde_json::Value =
            serde_json::from_str(&config_json).context("config.json is not JSON")?;
        match raw.get("vision_config") {
            Some(_) => Some(
                crate::dsv41::vision::VisionConfig::from_config_json(&raw)
                    .context("deepseek_v41 vision_config")?,
            ),
            None => None,
        }
    } else {
        None
    };
    // Every load, so a swap away from deepseek_v41 turns its bans off again.
    crate::dsv41::repetition::configure(dsv41);
    crate::dsv41::set_serving(dsv41);
    if dsv41 {
        // Resolve the JPEG decoder now so its version (or the fallback) is in the startup log.
        let _ = crate::dsv41::turbojpeg::available();
    }

    // ATLAS_SAMPLER_DEVICE_TOPK=1: receipt, or refuse to serve without the kernel.
    scheduler::device_topk_startup_receipt(&*scheduler_model)?;
    // ATLAS_QWEN4_TEXT_ONLY=1 did not load the vision tower: evaluate before
    // spawning the scheduler so a malformed env value fails before any detached
    // GPU-owning thread exists.
    let vision_config = if serve_phases::qwen4_text_only(&config)? {
        None
    } else {
        config.vision.clone()
    };
    // Moved into the scheduler thread; `None` leaves the gate disarmed.
    let scheduler_mtp_gate = args.mtp_gate.clone();
    let scheduler_max_seq_len = args.max_seq_len;
    // Spawn only after every fallible load step above has succeeded: a `?`
    // after this point would return with the scheduler thread already owning
    // the model, detached, so its GPU memory could never be released — fatal
    // for a swap that fails half way on a unified-memory box.
    let scheduler_handle = std::thread::spawn(move || {
        super::scheduler_supervisor::run_supervised(atlas_core::fault::global(), move || {
            scheduler::run(
                scheduler_model,
                request_rx,
                scheduler_eos,
                max_batch_size,
                use_speculative,
                num_drafts,
                policy,
                max_prefill_tokens,
                max_batch_tokens,
                prefill_chunk_with_active_decode,
                args.decode_gap_target_ms,
                use_self_spec,
                use_ngram_spec,
                swap_space_gb,
                high_speed_swap_cfg,
                block_size,
                scheduler_max_seq_len,
                think_end_token,
                think_start_token,
                code_fence_token,
                tool_call_start_token,
                tool_call_end_token,
                reflection_suppress_ids,
                grammar_engine,
                adaptive_sampling,
                session_manager,
                scheduler_spontaneous_think_budget,
                scheduler_mtp_gate,
                speculation_forced_disabled,
            );
        });
    });

    // Republish the levers for THIS run, beside the spawn that created the
    // scheduler they describe. A send failure means the dashboard thread is
    // gone, which is not a load failure — the server runs headless from here.
    if let Some(tx) = tui_handles_tx {
        let levers: crate::tui::RunLevers =
            Arc::new(crate::scheduler::levers::SchedLevers::from_env());
        let _ = tx.send(crate::tui::RunHandles { levers });
    }

    // 8. Build app state
    let model_ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    // `carried` is a PARAMETER, not built here. These three outlive any model:
    // a swap that rebuilt them would silently drop stored conversations and
    // responses and reset every rate-limit bucket — no error, just a user
    // noticing their history is gone. Taking them as an argument is what makes
    // the compiler ask the swap for them.
    let response_store = carried.response_store.clone();
    let rate_limiter = carried.rate_limiter.clone();
    let conversation_store = carried.conversation_store.clone();
    serve_phases::log_response_store_audit(&response_store, &rate_limiter);
    let dump_writer = serve_phases::open_dump_writer(&args);
    let dflash = args.dflash;
    let serving =
        crate::api::flux_ui::resolved_serving_block(crate::api::flux_ui::ResolvedServing {
            max_batch_size,
            ingress_queue: args.max_num_seqs,
            max_seq_len: args.max_seq_len,
            routes: serving_routes,
            dflash,
            self_speculative: use_self_spec,
            ngram: use_ngram_spec,
            num_drafts,
            world_size,
        });
    let state = Arc::new(AppState {
        tokenizer,
        model_name,
        max_seq_len: args.max_seq_len,
        yarn_context: serve_phases::text_only_yarn_context(&config),
        max_batch_size,
        serving,
        request_tx,
        vision_config,
        dsv41,
        dsv41_vision,
        default_temperature,
        default_top_k,
        default_top_p,
        default_top_n_sigma,
        default_min_p,
        tool_call_parser,
        reasoning_parser: reasoning_parser_box,
        think_end_token_id: think_end_token,
        think_start_token_id: think_start_token,
        tool_max_tokens: args.tool_max_tokens,
        default_max_tokens,
        sampling_presets,
        tool_call_start_token_id: tool_call_start_token,
        auto_compact_threshold: args.auto_compact,
        model_ready: model_ready.clone(),
        request_timeout: args.request_timeout,
        // Behavior from MODEL.toml, embedded at build time; effective_context
        // is not yet embedded in TargetPtxSet and remains 0 (unset).
        effective_context: 0,
        behavior: app_behavior,
        disable_thinking: args.disable_thinking,
        disable_simhash_watchdog: args.disable_simhash_watchdog
            || !serve_phases::simhash_watchdog_env_enabled(),
        default_chat_template_kwargs,
        runtime: runtime_settings,
        response_store,
        rate_limiter,
        conversation_store,
        dump_writer,
        auth,
    });

    serve_phases::log_behavior_audit(&args, &ptx_set);

    // Ready is asserted HERE, not in the router: the router is built once at
    // boot and a swap never rebuilds it, so a `model_ready` set there would
    // stay true across a swap that failed and false for one that succeeded.
    // The listener is not yet up on the first boot, so nothing can observe
    // this early; on a swap it is the moment the model became serveable.
    model_ready.store(true, std::sync::atomic::Ordering::Relaxed);

    Ok(Some(Prepared {
        state,
        bind: args.bind,
        port: args.port,
        scheduler: scheduler_handle,
        gpu: gpu_handle,
        oom_watchdog,
    }))
}

/// Cap default-sized `oom_guard_mb` on discrete GPUs with `<= 48 GiB` VRAM
/// to `min(configured, max(512, total_vram_mb / 8))` so 16–32 GiB cards are
/// not starved of KV cache while UMA (GB10) and `> 48 GiB` discrete GPUs keep
/// their configured margin untouched.
fn effective_discrete_oom_guard_mb(
    configured_mb: usize,
    total_vram_bytes: Option<usize>,
    is_discrete: bool,
) -> usize {
    const MIB: usize = 1024 * 1024;
    const SMALL_VRAM_THRESHOLD_MB: usize = 48 * 1024;
    const MIN_GUARD_MB: usize = 512;
    if !is_discrete {
        return configured_mb;
    }
    let Some(total_bytes) = total_vram_bytes.filter(|&bytes| bytes > 0) else {
        return configured_mb;
    };
    let total_vram_mb = total_bytes / MIB;
    if total_vram_mb > SMALL_VRAM_THRESHOLD_MB {
        return configured_mb;
    }
    configured_mb.min((total_vram_mb / 8).max(MIN_GUARD_MB))
}

fn qwen4_aux_speculation_unsupported(
    self_speculative: bool,
    ngram_speculative: bool,
    is_qwen4: bool,
) -> bool {
    is_qwen4 && (self_speculative || ngram_speculative)
}

fn qwen4_native_loaded_serial_control_requested(
    native_speculative: bool,
    dflash: bool,
    self_speculative: bool,
    ngram_speculative: bool,
    is_qwen4: bool,
) -> bool {
    is_qwen4 && native_speculative && !dflash && !self_speculative && !ngram_speculative
}

#[cfg(test)]
mod qualification_tests {
    use super::{
        effective_discrete_oom_guard_mb, qwen4_aux_speculation_unsupported,
        qwen4_native_loaded_serial_control_requested,
    };

    #[test]
    fn discrete_oom_guard_caps_small_gpus_without_affecting_uma_or_large_gpus() {
        const GIB: usize = 1024 * 1024 * 1024;
        // UMA GB10 keeps configured 4096 MB unchanged.
        assert_eq!(
            effective_discrete_oom_guard_mb(4096, Some(16 * GIB), false),
            4096
        );
        assert_eq!(
            effective_discrete_oom_guard_mb(4096, Some(128 * GIB), false),
            4096
        );
        // Discrete <= 48 GiB caps to min(configured, max(512, total_mb / 8)).
        assert_eq!(
            effective_discrete_oom_guard_mb(4096, Some(16 * GIB), true),
            2048
        );
        assert_eq!(
            effective_discrete_oom_guard_mb(4096, Some(24 * GIB), true),
            3072
        );
        assert_eq!(
            effective_discrete_oom_guard_mb(4096, Some(32 * GIB), true),
            4096
        );
        assert_eq!(
            effective_discrete_oom_guard_mb(4096, Some(4 * GIB), true),
            512
        );
        assert_eq!(
            effective_discrete_oom_guard_mb(256, Some(16 * GIB), true),
            256
        );
        // Discrete > 48 GiB keeps configured value untouched.
        assert_eq!(
            effective_discrete_oom_guard_mb(4096, Some(72 * GIB), true),
            4096
        );
        assert_eq!(
            effective_discrete_oom_guard_mb(4096, Some(96 * GIB), true),
            4096
        );
    }

    #[test]
    fn flash_next_auxiliary_speculation_is_fail_closed() {
        assert!(qwen4_aux_speculation_unsupported(false, true, true));
        assert!(qwen4_aux_speculation_unsupported(true, false, true));
        assert!(!qwen4_aux_speculation_unsupported(false, false, true));
        assert!(!qwen4_aux_speculation_unsupported(true, true, false));
    }

    #[test]
    fn glm_target_store_skips_the_safetensors_quant_heuristic() {
        let source = include_str!("serve_load.rs");
        let guard = source
            .find("    if has_glm53_target_store {\n        // GLM's weights live in its own")
            .expect("GLM quant-format guard");
        let detect = source
            .find("spark_model::quant_format::detect_quant_format(&config, &store)")
            .expect("safetensors quant-format detection");
        let else_branch = source[guard..]
            .find("    } else {")
            .map(|at| guard + at)
            .unwrap();
        assert!(guard < else_branch && else_branch < detect);
    }

    #[test]
    fn only_flash_next_native_mtp_is_a_loaded_serial_control() {
        assert!(qwen4_native_loaded_serial_control_requested(
            true, false, false, false, true,
        ));
        for request in [
            (false, false, false, false, true),
            (true, true, false, false, true),
            (true, false, true, false, true),
            (true, false, false, true, true),
            (true, false, false, false, false),
        ] {
            assert!(!qwen4_native_loaded_serial_control_requested(
                request.0, request.1, request.2, request.3, request.4,
            ));
        }
    }
}
