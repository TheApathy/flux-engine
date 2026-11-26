// SPDX-License-Identifier: AGPL-3.0-only

//! CLI argument parsing.

/// Closed value sets for enumerated string flags.
#[path = "cli_flag_values.rs"]
pub(crate) mod flag_values;

#[path = "cli_dflash_environment.rs"]
mod dflash_environment;
#[path = "cli_dflash_lookup.rs"]
pub(crate) mod dflash_lookup;

/// Refuse values outside enumerated flags' closed sets.
///
/// Called both at CLI startup and by the TUI Library form via `Recipe::serve_args`
/// so invalid flag values are caught before weight loading. Every problem is
/// reported together rather than stopping at the first error.
pub(crate) fn validate_serve_args(args: &ServeArgs) -> Result<(), String> {
    let checks: [(&str, &str); 4] = [
        ("scheduling-policy", args.scheduling_policy.as_str()),
        ("kv-cache-dtype", args.kv_cache_dtype.as_str()),
        (
            "dflash-draft-sample-method",
            args.dflash_draft_sample_method.as_str(),
        ),
        ("role", args.role.as_str()),
    ];
    let mut problems = Vec::new();
    for (flag, value) in checks {
        let Some(allowed) = flag_values::options_for_flag(flag) else {
            continue;
        };
        if !allowed.iter().any(|a| a == value) {
            problems.push(format!(
                "--{flag} {value:?} is not one of: {}",
                allowed.join(", ")
            ));
        }
    }
    // ── Ranges ──
    // clap parses `--gpu-memory-utilization 9.0` happily: it is a valid f64.
    // The number is a FRACTION, so anything above 1.0 asks for more memory
    // than the device has and fails during allocation, minutes in.
    if !(0.0..=1.0).contains(&args.gpu_memory_utilization) {
        problems.push(format!(
            "--gpu-memory-utilization {} is a fraction of device memory; it must be in 0.0..=1.0",
            args.gpu_memory_utilization
        ));
    }
    if args.max_batch_size == 0 {
        problems.push("--max-batch-size must be at least 1".to_string());
    } else if args.max_batch_size > spark_runtime::buffers::DECODE_META_MAX_ROWS {
        problems.push(format!(
            "--max-batch-size {} exceeds the qualified decode-metadata ceiling of {}",
            args.max_batch_size,
            spark_runtime::buffers::DECODE_META_MAX_ROWS,
        ));
    }
    if args.max_num_seqs == 0 {
        problems.push("--max-num-seqs must be at least 1".to_string());
    }
    if args.kv_pool_tokens == Some(0) {
        problems
            .push("--kv-pool-tokens must be positive (omit it for the per-slot pool)".to_string());
    }
    if args.kv_pool_tokens.is_some() && args.high_speed_swap {
        problems.push(
            "--kv-pool-tokens cannot be combined with --high-speed-swap, which sizes its own \
             per-sequence HBM window"
                .to_string(),
        );
    }
    if args.max_queue_depth == Some(0) {
        problems.push(
            "--max-queue-depth must be at least 1 (omit it for the default queue)".to_string(),
        );
    }
    if args.pp_size == 0 {
        problems.push("--pp-size must be at least 1".to_string());
    }
    if let Some(peer) = args.disagg_peer.as_deref() {
        if peer.trim().is_empty() {
            problems.push("--disagg-peer must not be empty".to_string());
        }
        if args.role == "prefill" {
            problems.push(
                "--disagg-peer cannot be used with --role prefill; pass it on --role decode or --role unified"
                    .to_string(),
            );
        }
    }

    // ── Cross-flag ──
    // The pair, not the field: `--num-drafts` past 1 asks the scheduler to
    // verify drafts that nothing produces unless a speculative method is on.
    // A per-field check cannot see this, which is the whole reason the form
    // validates the WHOLE config after every edit rather than the one row.
    let any_speculative =
        args.speculative || args.self_speculative || args.ngram_speculative || args.dflash;
    if args.num_drafts > 1 && !any_speculative {
        problems.push(format!(
            "--num-drafts {} needs a speculative method: pass one of --speculative, \
             --self-speculative, --ngram-speculative or --dflash",
            args.num_drafts
        ));
    }

    // Self-spec and n-gram own separate scheduler state machines. They cannot
    // be composed with each other or with target-MTP/DFlash: the scheduler's
    // precedence would silently shadow the later method while still loading
    // and advertising it. The one supported composition is native target MTP
    // as DFlash's explicitly constructed alternate proposer arm.
    let auxiliary_methods: Vec<&str> = [
        (args.self_speculative, "--self-speculative"),
        (args.ngram_speculative, "--ngram-speculative"),
    ]
    .into_iter()
    .filter_map(|(enabled, flag)| enabled.then_some(flag))
    .collect();
    if !auxiliary_methods.is_empty()
        && (auxiliary_methods.len() > 1 || args.speculative || args.dflash)
    {
        problems.push(format!(
            "{} cannot be combined with another speculative method; scheduler precedence would \
             silently bypass one mode (--dflash plus --speculative is the sole supported dual-arm combination)",
            auxiliary_methods.join(" and ")
        ));
    }

    // DFlash is a correctness-sensitive integration: accepting a typo and
    // silently substituting BF16 makes the advertised profile false, while
    // prefix-cache reuse is known to produce incorrect multi-turn results on
    // SM12.x. Reject both before loading many gigabytes of weights.
    if !matches!(args.dflash_quantization.as_str(), "bf16" | "nvfp4") {
        problems.push(format!(
            "--dflash-quantization {:?} is not one of: bf16, nvfp4",
            args.dflash_quantization
        ));
    }
    let probabilistic_dflash = args.dflash_draft_sample_method == "probabilistic";
    if probabilistic_dflash && !args.dflash {
        problems.push(
            "--dflash-draft-sample-method probabilistic requires --dflash and would otherwise \
             be an inert sampling claim"
                .to_string(),
        );
    }
    if probabilistic_dflash && !args.experimental_dflash_probabilistic {
        problems.push(
            "--dflash-draft-sample-method probabilistic is default-off and requires the explicit \
             --experimental-dflash-probabilistic qualification flag"
                .to_string(),
        );
    }
    if args.experimental_dflash_probabilistic && !(args.dflash && probabilistic_dflash) {
        problems.push(
            "--experimental-dflash-probabilistic requires the exact pair --dflash \
             --dflash-draft-sample-method probabilistic; refusing an inert qualification claim"
                .to_string(),
        );
    }
    if probabilistic_dflash && (args.speculative || args.mtp_gate.is_some()) {
        problems.push(
            "probabilistic DFlash cannot be combined with --speculative or --mtp-gate: the MTP \
             arm does not produce the sparse-q frame required by exact p/q acceptance"
                .to_string(),
        );
    }
    if args.dflash && args.enable_prefix_caching {
        problems.push(
            "--dflash cannot be combined with --enable-prefix-caching: the combination has a \
             known SM12.x correctness regression on multi-turn cache hits"
                .to_string(),
        );
    }
    // Topology normalization runs after this admission gate and can derive
    // world_size from tp_size. DFlash is qualified for single-rank and
    // lockstep TP=2/4 (with EP=1 or EP=TP, PP=1); PP>1 and other distributed
    // DFlash topologies remain fail-closed.
    let dflash_topology_admitted =
        (args.world_size == 1 && args.tp_size == 1 && args.ep_size == 1 && args.pp_size == 1)
            || (matches!(args.tp_size, 2 | 4)
                && (args.ep_size == 1 || args.ep_size == args.tp_size)
                && args.pp_size == 1
                && (args.world_size == 1 || args.world_size == args.tp_size));
    if args.dflash && !dflash_topology_admitted {
        problems.push(format!(
            "--dflash supports single-rank or TP=2/4 (EP=1 or EP=TP, PP=1) only: world_size={}, \
             tp_size={}, ep_size={}, pp_size={}; other distributed DFlash topologies are not \
             qualified",
            args.world_size, args.tp_size, args.ep_size, args.pp_size
        ));
    }
    // The auxiliary paths do not implement a complete cross-rank command,
    // checkpoint, rollback, and abort protocol. In particular self-spec has
    // no draft/verify worker protocol, while n-gram cannot broadcast a policy
    // walk failure after K=2 verify. Keep both single-rank until that protocol
    // is implemented and qualified.
    if (args.self_speculative || args.ngram_speculative)
        && (args.world_size != 1 || args.tp_size != 1 || args.ep_size != 1 || args.pp_size != 1)
    {
        let mode = if args.self_speculative {
            "--self-speculative"
        } else {
            "--ngram-speculative"
        };
        problems.push(format!(
            "{mode} is single-rank only: world_size={}, tp_size={}, ep_size={}, pp_size={}; distributed \
             checkpoint/rollback execution is not qualified",
            args.world_size, args.tp_size, args.ep_size, args.pp_size
        ));
    }

    // The DDTree implementation still has several error branches that emit a
    // flat-chain payload and continue.  Until those branches are made
    // fail-closed, accepting the method or its tuning fields would advertise
    // a runtime configuration the engine cannot prove it executed.
    if args.dflash_method != "flat" {
        problems.push(format!(
            "--dflash-method {:?} is unavailable in this release; only `flat` is supported. \
             DDTree can currently fall back to flat at runtime, so Flux must fail closed",
            args.dflash_method
        ));
    }
    let mut unwired_ddtree_fields = Vec::new();
    if args.ddtree_budget.is_some() {
        unwired_ddtree_fields.push("--ddtree-budget");
    }
    if args.ddtree_top_k != 8 {
        unwired_ddtree_fields.push("--ddtree-top-k");
    }
    if args.ddtree_temperature.to_bits() != 1.0f32.to_bits() {
        unwired_ddtree_fields.push("--ddtree-temperature");
    }
    if !args.ddtree_chain_seed {
        unwired_ddtree_fields.push("--ddtree-chain-seed");
    }
    if !unwired_ddtree_fields.is_empty() {
        problems.push(format!(
            "{} cannot be used while DDTree is unavailable; refusing fields that would have no \
             authoritative runtime effect",
            unwired_ddtree_fields.join(", ")
        ));
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; "))
    }
}

/// Validate the live environment against the release-flat DFlash contract.
pub(crate) fn validate_dflash_runtime_environment(args: &ServeArgs) -> Result<(), String> {
    validate_dflash_runtime_environment_with(args, dflash_environment_value)
}

/// Preserve the distinction between absent and non-Unicode environment
/// values. Profile application uses `var_os` to decide whether an operator
/// supplied a value, so treating `VarError::NotUnicode` as absent here would
/// let that value suppress a safe profile default without being validated.
pub(crate) fn dflash_environment_value(name: &str) -> Option<String> {
    dflash_environment::live_environment_value(name)
}

pub(crate) fn validate_dflash_runtime_environment_with(
    args: &ServeArgs,
    get: impl FnMut(&str) -> Option<String>,
) -> Result<(), String> {
    dflash_environment::validate_flat_dflash_environment(
        dflash_environment::FlatDflashContract {
            enabled: args.dflash,
            method: &args.dflash_method,
            gamma: args.dflash_gamma,
            window_size: args.dflash_window_size,
            quantization: &args.dflash_quantization,
        },
        get,
    )
}

use clap::Parser;

/// The crate version, exposed for the CLI banner, TUI help tab, and issue reporter.
pub const FLUX_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const ATLAS_VERSION: &str = env!("CARGO_PKG_VERSION");
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "flux-server",
    version = FLUX_VERSION,
    about = "Flux Engine — High-Velocity Blackwell SM121a LLM Inference Server (derived from Atlas)"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Start the inference server.
    Serve(ServeArgs),
}

/// Top-level binary CLI (`main.rs`), adding host launcher subcommands while
/// keeping [`Cli`] / [`Command`] irrefutably `Serve`-only for `Recipe` and
/// serve-arg tests.
#[derive(Parser, Debug)]
#[command(
    name = "flux-server",
    version = FLUX_VERSION,
    about = "Flux Engine — High-Velocity Blackwell SM121a LLM Inference Server (derived from Atlas)"
)]
pub struct EntryCli {
    #[command(subcommand)]
    pub command: EntryCommand,
}

#[derive(clap::Subcommand, Debug)]
pub enum EntryCommand {
    /// Start the inference server.
    Serve(ServeArgs),
    /// Run a command (`flux-server serve ...`) with the GB10 display carveout as
    /// extra KV memory.
    DisplayCarveout(crate::display_carveout::DisplayCarveoutArgs),
}

/// Arguments for the `serve` subcommand.
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ServeArgs {
    /// HuggingFace model ID (e.g. "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4")
    /// or a local directory path containing config.json.
    #[arg(value_name = "MODEL", required_unless_present = "model_from_path")]
    pub model: Option<String>,

    /// Load model directly from this filesystem path (skips HF cache resolution).
    #[arg(long, value_name = "PATH")]
    pub model_from_path: Option<PathBuf>,

    /// Fail-closed recipe contract (`recipe.toml`). Startup refuses when the
    /// checkpoint, env gates or serve flags differ from it, and logs one
    /// `RECIPE_RECEIPT` line otherwise. Also read from `FLUX_RECIPE`; this flag
    /// wins. Absent both, nothing is checked (legacy mode).
    #[arg(long, value_name = "PATH")]
    pub recipe: Option<PathBuf>,

    /// Override model name shown in /v1/models and API responses.
    /// Defaults to the positional MODEL argument, then config.json _name_or_path.
    #[arg(long, alias = "served-model-name", value_name = "NAME")]
    pub model_name: Option<String>,

    /// Force selection of a specific kernel target by directory name
    /// (e.g. `qwen3.6-35b-a3b-abl`). Bypasses the
    /// `(model_type, hidden_size)` first-match-wins dispatch in
    /// `atlas_kernels::ptx_for_config`. Use this to apply tuned MODEL.toml
    /// defaults (sampling, behavior) for abliterated / heretic checkpoints
    /// that share their architecture with a canonical target — without
    /// this flag, the canonical target's MODEL.toml wins because both
    /// match the same `(model_type, hidden_size)` tuple.
    ///
    /// The named target must exist as a compiled `(hw, model, quant)`
    /// kernel set; the build only produces it when
    /// `ATLAS_TARGET_MODEL=*` (wildcard), the exact name, or a strict
    /// comma-separated list containing the exact name is set at build time.
    /// Errors with the list of available targets if the requested name does
    /// not match.
    #[arg(long, value_name = "MODEL_DIR")]
    pub kernel_target: Option<String>,

    /// Resolve all kernels for the model and exit, reporting any that did not
    /// resolve. Does not start the server. The EXIT CODE IS THE NUMBER of
    /// unresolved kernels — 0 means every lookup resolved.
    ///
    /// A dry run that stops immediately after the kernel audit: config, GPU
    /// init, weight load and model construction all run (every lookup lives in
    /// a layer constructor, so a check that skipped them would resolve a
    /// DIFFERENT set than a real serve), then the report is printed and the
    /// process exits. The scheduler is never started and no port is bound.
    ///
    /// A POSIX status is 8 bits, so the code is CLAMPED at 255 — 256 would be
    /// reported as 0, i.e. a broken model reading as a pass. Whenever the clamp
    /// bites, the true count is printed alongside it and carried in the JSON.
    ///
    /// A one-line JSON object (`{"atlas_kernel_check": …}`) is printed on
    /// stdout after the human report, so a sweep over every target can
    /// aggregate without parsing prose.
    ///
    /// This flag is diagnostic-only and does not alter normal `serve` startup.
    #[arg(long, default_value_t = false)]
    pub check_kernels: bool,

    /// Disable the interactive dashboard and retain plain log output.
    ///
    /// Plain mode is also selected automatically when stdout is not a TTY.
    #[arg(long, default_value_t = false)]
    pub no_tui: bool,

    /// Override HuggingFace cache directory
    /// (default: $HF_HUB_CACHE, $HF_HOME/hub, or ~/.cache/huggingface/hub).
    #[arg(long, value_name = "DIR")]
    pub cache_dir: Option<PathBuf>,

    /// HTTP port.
    #[arg(long, default_value_t = 8888)]
    pub port: u16,

    /// GPU ordinal.
    #[arg(long, default_value_t = 0)]
    pub gpu_ordinal: usize,

    /// Maximum sequence length.
    #[arg(long, default_value_t = 32768)]
    pub max_seq_len: usize,

    /// Explicit RoPE theta for contexts beyond the checkpoint's native
    /// max_position_embeddings. Required for an experimental extension;
    /// capacity does not imply long-context quality.
    #[arg(long)]
    pub rope_theta_override: Option<f64>,

    /// Static YaRN factor for an explicitly extended context. Qwen3.8's
    /// official 1M recipe uses 4 and keeps rope_theta unchanged.
    #[arg(long)]
    pub rope_yarn_factor: Option<f32>,
    /// Enable Qwen4 QSA beyond 2048 tokens. The initial exact path requires
    /// C=1 and BF16 KV; unsupported combinations fail startup.
    #[arg(long, default_value_t = false)]
    pub qwen4_qsa: bool,

    /// KV cache block size (tokens per block).
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// KV cache dtype (fp8, bf16, or nvfp4).
    /// Default: fp8. NVFP4 uses less memory but may lose coherence at long context
    /// without --kv-high-precision-layers. FP8 is the safe default.
    #[arg(long, default_value = "fp8")]
    pub kv_cache_dtype: String,

    /// Boundary attention layers to keep at BF16 KV cache precision (first N + last N).
    /// Protects attention sink tokens (early layers) and output quality (final layers)
    /// from quantization error while saving memory on middle layers.
    /// Accepts: number, "auto" (=2, recommended), "max"/"all" (all BF16).
    /// Default: 0 (all layers use --kv-cache-dtype).
    #[arg(long, default_value = "0")]
    pub kv_high_precision_layers: String,

    /// Explicit set of attention-layer indices to keep at BF16 KV precision,
    /// from a per-layer KV quantization sensitivity sweep. Overrides the
    /// positional `--kv-high-precision-layers` heuristic when set. Indices are
    /// attention-layer-local (`0..num_attention_layers`, in attention order —
    /// not global model layer IDs). Comma-separated, e.g. `"0,4,9,15"`.
    /// Empty uses the positional heuristic.
    #[arg(long, default_value = "")]
    pub kv_high_precision_layer_set: String,

    /// GPU memory utilization (0.0-1.0).
    #[arg(long, default_value_t = 0.90)]
    pub gpu_memory_utilization: f64,

    /// CUDA runtime headroom reserved when speculative decoding is enabled.
    /// The conservative default protects ordinary checkpoints from transient
    /// conversion/scratch allocations. Memory-tight unified-memory systems
    /// may lower this after all draft weights are already materialized.
    #[arg(long, default_value_t = 4096)]
    pub speculative_cuda_headroom_mb: usize,

    /// Maximum concurrent sequences.
    #[arg(long, default_value_t = 128)]
    pub max_num_seqs: usize,

    /// Global kill-switch for chain-of-thought / reasoning output.
    /// When set, the server forces thinking OFF regardless of what the
    /// client requests (reasoning_effort, thinking.budget_tokens, etc.)
    /// or what MODEL.toml declares as the default. Precedence (highest
    /// wins): this flag → request body → MODEL.toml `[behavior]`.thinking_default.
    #[arg(long, default_value_t = false)]
    pub disable_thinking: bool,

    /// Disable content-loop watchdogs for workloads where repeated structure is
    /// legitimate (for example long HTML/CSS/JavaScript output). This overrides
    /// both MODEL.toml's configured detector and the catastrophic repetition
    /// guard. It does not change sampling.
    #[arg(long, default_value_t = false)]
    pub disable_loop_watchdog: bool,

    /// Disable the thinking-loop watchdog. This is separate from the content
    /// watchdog because repeated reasoning can be legitimate on long math and
    /// code tasks. The explicit thinking-token budget remains enforced.
    #[arg(long, default_value_t = false)]
    pub disable_thinking_loop_watchdog: bool,

    /// Disable MODEL.toml's confidence-based early stop for reasoning.
    /// This leaves the explicit thinking-token budget and both loop watchdogs
    /// unchanged. Intended for controlled quality evaluation of cases where
    /// early stopping may force a premature answer transition.
    #[arg(long, default_value_t = false)]
    pub disable_confidence_early_stop: bool,

    /// Disable the API streaming SimHash semantic-loop watchdog. This leaves
    /// the scheduler content/thinking watchdogs unchanged. Intended for
    /// controlled evaluation of long mathematical or structured answers where
    /// legitimately repeated derivation language can resemble a paraphrase
    /// loop.
    #[arg(long, default_value_t = false)]
    pub disable_simhash_watchdog: bool,

    /// Tokenize special-token text inside chat message content and tool
    /// definitions as the special tokens themselves (the pre-0.9.15.6
    /// behaviour), for clients that deliberately splice turns. Off: such text
    /// is literal and only the chat template's own markers are special. Also
    /// enabled by FLUX_ALLOW_SPECIAL_TOKENS_IN_CONTENT=1.
    #[arg(long, default_value_t = false)]
    pub allow_special_tokens_in_content: bool,

    /// Override MODEL.toml's `[behavior].max_thinking_budget` (tokens).
    /// Sets the per-request ceiling for thinking-block length. Per-request
    /// `thinking.budget_tokens` (or `reasoning_effort`) still wins below
    /// this ceiling; the (max_tokens * 9 / 10) safety cap is always enforced.
    #[arg(long, alias = "thinking-budget")]
    pub max_thinking_budget: Option<u32>,

    /// Reserve N tokens for chain-of-thought reasoning ON TOP OF `max_tokens`
    /// (clamped to remaining context window `max_seq_len - prompt_len`) while
    /// capping `thinking_budget <= N`, so reasoning never starves visible
    /// JSON/code output. Also configurable via `FLUX_THINKING_RESERVE`.
    #[arg(long, value_name = "TOKENS")]
    pub thinking_reserve: Option<u32>,

    /// Cap spontaneous or template-forced `<think>` blocks at N tokens when
    /// `enable_thinking == false` (e.g. `256`). Also configurable via
    /// `FLUX_THINKING_OFF_BUDGET`.
    #[arg(long, value_name = "TOKENS")]
    pub thinking_off_budget: Option<u32>,

    /// Override the MODEL.toml inter-tool prose cap. Precedence is CLI,
    /// then `ATLAS_MAX_INTER_TOOL_PROSE`, then MODEL.toml/default. Zero
    /// disables this guard.
    #[arg(long)]
    pub max_inter_tool_prose: Option<u32>,

    /// Default chat template kwargs applied when the client sends no
    /// thinking parameters (no `reasoning.effort`, `chat_template_kwargs`,
    /// or `enable_thinking` in the request body). A JSON object with
    /// optional keys: `enable_thinking` (bool), `thinking_budget` (u32),
    /// and `preserve_thinking` (bool, for checkpoint templates that support it).
    ///
    /// Precedence (highest wins): request body → this flag → MODEL.toml.
    /// Example: `--default-chat-template-kwargs '{"enable_thinking":true}'`
    #[arg(long, value_name = "JSON")]
    pub default_chat_template_kwargs: Option<String>,

    /// Currently slower than regular decode for hybrid SSM models.
    #[arg(long, default_value_t = false)]
    pub speculative: bool,

    /// Supplemental Flash-Next native-MTP checkpoint. Indexed sidecars may
    /// reference absent base-model shards; Flux loads only `mtp.*` tensors.
    #[arg(long, value_name = "PATH")]
    pub mtp_from_path: Option<std::path::PathBuf>,

    /// Enable self-speculative decoding: draft via layer-skipping (no MTP weights needed).
    /// Skips SSM layers during drafting for cheap predictions, then verifies with full model.
    #[arg(long, default_value_t = false)]
    pub self_speculative: bool,

    /// Enable N-gram speculative decoding: CPU-side pattern matching proposer
    /// with CUDA-graphed K=2 verification. No extra weights needed.
    #[arg(long, default_value_t = false)]
    pub ngram_speculative: bool,

    /// Enable DFlash block-diffusion speculative decoding (Z Lab,
    /// arXiv 2602.06036). Pairs the target with a small Qwen3-architecture
    /// drafter (e.g. `z-lab/Qwen3.6-35B-A3B-DFlash`) that emits γ tokens
    /// per step via bidirectional in-block attention conditioned on captured
    /// target hidden states.
    ///
    /// May be combined with `--speculative`: the DFlash drafter is installed
    /// as proposer arm 0 and the target's in-checkpoint MTP head is kept on
    /// arm 1 instead of being discarded, so `--mtp-gate auto` can arbitrate
    /// between them per step. The two heads have opposite strengths on this
    /// model family — the external drafter wins on code, the native head on
    /// prose. Passing both costs one extra resident MTP head (~0.78 GB at the
    /// default `--mtp-quantization bf16`, ~0.24 GB at nvfp4).
    ///
    /// These flags used to be `conflicts_with` each other. They no longer are:
    /// the exclusion was a CLI convention, not an engine constraint — a
    /// `--dflash` run already built the MTP head and threw it away.
    #[arg(long, default_value_t = false)]
    pub dflash: bool,

    /// Speculation throughput gate: `auto`, `force`, `dflash`, or `mtp`.
    ///
    /// Omitted (the DEFAULT) means NO gate — speculation runs exactly as
    /// configured, byte-for-byte as it did before the gate existed. This is
    /// deliberately not a clap default so that the champion configuration
    /// stays the control in any A/B.
    ///
    /// `auto` arms the arbiter, which measures DELIVERED tok/s per arm over
    /// 16-step windows and runs whichever arm is fastest right now, with
    /// hysteresis, a dwell requirement, and periodic probing of the others.
    /// It measures throughput rather than draft acceptance on purpose:
    /// acceptance is expressed in each arm's own units (a mean of 3 is winning
    /// for K=2 and losing badly for γ=16), so no acceptance threshold orders
    /// the arms correctly.
    ///
    /// `force` DISARMS the arbiter so verify keeps flowing even where it would
    /// measure net-negative — a diagnostic for collecting verify samples, and
    /// never a production setting: if forcing wins, the gate is miscalibrated
    /// and that is the fix. To run without speculation at all, omit
    /// `--speculative` and `--dflash`.
    ///
    /// `dflash` / `mtp` PIN one proposer arm for the whole run. They exist so
    /// the two arms can be A/B'd on one binary and one set of weights;
    /// `mtp` requires a build that passed both `--dflash` and `--speculative`.
    #[arg(long)]
    pub mtp_gate: Option<String>,

    /// HuggingFace id (or local path) of the DFlash drafter checkpoint.
    /// When `--dflash` is set without `--draft-model`, the value falls
    /// through from the target's MODEL.toml `[dflash].draft_model` field.
    #[arg(long)]
    pub draft_model: Option<String>,

    /// Dense target checkpoint that donated the embedding and LM head used
    /// when training a DFlash drafter. Required for the experimental
    /// Qwen3.8-Flash-Next bridge because Flash-Next is 2560-wide while the
    /// existing v3 drafter shares 5120-wide dense-target tensors.
    #[arg(long)]
    pub dflash_donor_model: Option<String>,

    /// DFlash/DSpark draft count γ (parallel speculative tokens per step).
    /// Width semantics are checkpoint-family specific: legacy DFlash counts
    /// one known anchor (`B16 -> γ15`), while published DSpark counts draft
    /// rows (`B7 -> γ7`, verified as K8 with the target bonus). Smaller widths
    /// remain valid performance/cost tradeoffs. When omitted, Flux derives the
    /// trained width from config.
    #[arg(long)]
    pub dflash_gamma: Option<usize>,

    /// GLM-5.3 DFlash2 only: drafts the target verifies per step (1..=7).
    /// The checkpoint still proposes its trained block of 8 (`--dflash-gamma
    /// 8`); this bounds the verify width, which is the step's cost. Size
    /// `ATLAS_GLM53_PREFIX_COMMIT_ROWS` to at least this value.
    #[arg(long)]
    pub glm_dflash_max_drafts: Option<u8>,

    /// DSpark target-verify planner: `static`, `cap-accept`, or `compact`.
    /// Only `static` is currently supported. It verifies the full gamma+1
    /// window and deliberately skips confidence scoring, matching SGLang PR
    /// 34966's default. The dynamic modes are recognized but fail startup
    /// until Flux implements ragged verify layouts and SPS-based costing.
    #[arg(long, default_value = "static", value_name = "MODE")]
    pub dspark_verify_mode: String,

    /// DFlash drafter sliding-window size for long context. The generic
    /// upstream sglang / vLLM default is 4096; set to 0 to disable it (full
    /// attention). GLM-5.3 DFlash2 maps this inherited default to its
    /// checkpoint-pinned 2048-token window, also accepts an explicit 2048,
    /// and rejects every other override rather than silently ignoring it.
    #[arg(long, default_value_t = 4096)]
    pub dflash_window_size: usize,

    /// DFlash drafter weight precision: `bf16` or `nvfp4`. `bf16` loads
    /// weights verbatim from the drafter checkpoint and uses BF16 dense
    /// GEMMs (~134 ms/step). `nvfp4` runtime-quantizes every dense
    /// projection (7/layer + `fc`) to NVFP4 at model-load time so the
    /// per-step forward runs through `w4a16_gemm` kernels (~3-5x faster
    /// on GB10). Default `bf16` to preserve existing behavior.
    #[arg(long, default_value = "bf16")]
    pub dflash_quantization: String,

    /// DFlash draft-token selection: `greedy` preserves the released CUDA
    /// selector. `probabilistic` uses sampled DFlash2 sparse-q proposals and
    /// exact target p/q acceptance. It remains default-off and also requires
    /// `--experimental-dflash-probabilistic`.
    #[arg(long, default_value = "greedy")]
    pub dflash_draft_sample_method: String,

    /// Admit the narrow sampled-DFlash2 qualification path. This flag has no
    /// effect by itself: it requires `--dflash --dflash-draft-sample-method
    /// probabilistic`, single-rank TP1/EP1, a DFlash2 checkpoint with its
    /// candidate selector, and per-request positive temperature plus a seed.
    /// Grammar, tools, logprobs, adaptive sampling, alternate proposer arms,
    /// trees, retrieval, recycling, and batching remain HOLD. Thinking is
    /// admitted only through the exact live-phase/policy-width gates; forced
    /// and one-token transition boundaries stay on ordinary decode.
    #[arg(long, default_value_t = false)]
    pub experimental_dflash_probabilistic: bool,

    /// DFlash verifier strategy. `flat` is the only release-supported method.
    /// `ddtree` is rejected until every tree failure path fails closed instead
    /// of emitting a flat-chain payload.
    #[arg(long, default_value = "flat")]
    pub dflash_method: String,

    /// Reserved DDTree node budget. Any non-default value is rejected while
    /// DDTree is unavailable.
    #[arg(long)]
    pub ddtree_budget: Option<usize>,

    /// Reserved DDTree sibling count. Only the inert default is accepted while
    /// DDTree is unavailable.
    #[arg(long, default_value_t = 8)]
    pub ddtree_top_k: usize,

    /// Reserved DDTree draft temperature. Only the inert default is accepted
    /// while DDTree is unavailable.
    #[arg(long, default_value_t = 1.0)]
    pub ddtree_temperature: f32,

    /// Reserved DDTree chain seed. Only the inert default is accepted while
    /// DDTree is unavailable.
    #[arg(long, default_value_t = true)]
    pub ddtree_chain_seed: bool,

    /// Number of draft tokens per speculative step (1=K=2, 2=K=3, 3=K=4 verify).
    /// Higher K verifies more drafts per step. Uses WY-chunkwise GDN kernels.
    #[arg(long, default_value_t = 1)]
    pub num_drafts: usize,

    /// Maximum concurrent sequences batched into one GPU decode step.
    #[arg(long, default_value_t = 8)]
    pub max_batch_size: usize,

    /// Shared paged-KV pool size in tokens. Unset (the default), the pool is
    /// sized so every one of --max-batch-size slots can reach --max-seq-len at
    /// once and startup refuses anything smaller. Set, the pool holds this many
    /// tokens (plus one spare block per slot and the dummy block), startup
    /// only requires one full --max-seq-len sequence to fit, and depth-aware
    /// admission queues requests whose reservation does not fit. Example:
    /// 2 x 512K sessions at C4 is --max-seq-len 524288 --max-batch-size 4
    /// --kv-pool-tokens 1048576. Launcher: FLUX_KV_POOL_TOKENS.
    #[arg(long)]
    pub kv_pool_tokens: Option<usize>,

    /// Requests that may wait for a slot or for KV capacity before new ones
    /// are refused with 429. Unset keeps the default bounded queue (the
    /// --max-num-seqs channel plus the scheduler's pending queue). Running
    /// requests are bounded by --max-batch-size. Launcher: FLUX_QUEUE_DEPTH.
    #[arg(long)]
    pub max_queue_depth: Option<usize>,

    /// MTP head weight precision: bf16 (default, highest acceptance rate
    /// = highest end-to-end throughput; the MTP head is small so the memory
    /// cost is modest), fp8 (1 byte/weight, balanced; slower draft due to
    /// a D2H sync in MoE dispatch), nvfp4 (0.5 byte/weight, fastest draft
    /// forward but lossier projections → lower acceptance rate, so end-to-
    /// end throughput is usually worse than bf16).
    #[arg(long, default_value = "bf16")]
    pub mtp_quantization: String,

    /// MTP draft vocabulary size. Limits the LM head GEMV to the first N
    /// token IDs, reducing propose latency. BPE tokenizers place frequent
    /// tokens at low IDs — 100K covers >99% of English outputs while
    /// cutting propose time by 37% (2.15ms → 1.35ms) with zero acceptance
    /// loss. Set to 0 to use full vocabulary.
    #[arg(long, default_value_t = 100000)]
    pub mtp_vocab: u32,

    /// Enable prefix caching via radix tree (RadixAttention).
    /// Caches KV blocks for recurring prompt prefixes. For SSM models,
    /// KV is recomputed when no SSM snapshot exists (safe but no TTFT speedup
    /// without Marconi snapshots). Block table reuse still avoids allocation.
    #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true")]
    pub enable_prefix_caching: bool,

    /// Dump every /v1/chat/completions, /v1/responses, and
    /// /v1/messages (Anthropic) request — plus the corresponding
    /// response (non-streaming) or aggregated stream — as JSONL to a
    /// file. Intended for extracting the exact system prompt and tool
    /// schema a client (opencode, Claude Code, etc.) is sending, and
    /// for replaying failure cases in fixtures.
    ///
    /// With no value: a temp file is created under $TMPDIR and its
    /// path is logged at INFO on startup. With a PATH: appends (never
    /// truncates) to that file. Each line is one JSON object:
    ///   `{ "ts": "<iso8601>", "endpoint": "...", "kind": "request"|"response",`
    ///     "seq": N, "body": { ... } }
    /// so entries can be grouped by `seq` to reconstruct pairs.
    #[arg(long, num_args = 0..=1, default_missing_value = "<auto>", value_name = "PATH")]
    pub dump: Option<String>,

    /// Scheduling policy: fifo (default) or slai (SLO-aware).
    /// SLAI prioritizes decode for sequences nearing TBT deadline
    /// and orders prefills shortest-prompt-first.
    #[arg(long, default_value = "fifo")]
    pub scheduling_policy: String,

    /// TBT deadline in milliseconds for SLAI scheduling policy.
    /// Sequences approaching this deadline trigger decode-first priority.
    #[arg(long, default_value_t = 100)]
    pub tbt_deadline_ms: u64,

    /// Maximum tokens to prefill per scheduler iteration (chunked prefill).
    /// Long prompts are split into chunks of this size, interleaved with
    /// decode steps for active sequences. 8192 is the default chunk size;
    /// set to 0 to disable chunking (process entire prompt in one shot).
    #[arg(long, default_value_t = 8192)]
    pub max_prefill_tokens: usize,

    /// Prefill chunk cap (tokens) used while at least one other sequence is
    /// decoding. Each scheduler tick runs one prefill chunk and then one
    /// decode step that waits for it, so every active decode stream stalls
    /// for one chunk per tick; this cap bounds that stall without shrinking
    /// the chunk an idle server uses for TTFT. 0 (default) disables the cap.
    /// Otherwise a positive multiple of 16 below --max-prefill-tokens.
    /// Capped requests are prefilled with more chunks, so their logits are
    /// not byte-identical to an uncapped run, and on Qwen4 a cap below
    /// ATLAS_QWEN4_PREFILL_WIDE sends those requests to the ordinary
    /// (slower) prefill path. Recipe key: prefill_chunk_with_active_decode.
    #[arg(long, default_value_t = 0, value_name = "TOKENS")]
    pub prefill_chunk_with_active_decode: usize,

    /// Decode inter-token gap target (milliseconds) for the SLO-budgeted
    /// prefill interleave. While at least one sequence decodes, each tick
    /// runs at most one prefill chunk: the largest prepared fast-prefill
    /// width (Qwen4 with ATLAS_QWEN4_PREFILL_WIDE + MULTI_CHUNK: powers of two
    /// from 512 up to the wide width, aligned to the prompt offset) whose
    /// predicted time fits the target minus the measured decode step. A
    /// prompt whose remainder fits one chunk predicted within 2x the target
    /// runs whole; a prompt that gets no chunk for floor(target / decode
    /// step) steps runs its smallest aligned width. Predictions are learned
    /// per (width, prompt-offset band) from measured chunks
    /// (atlas_prefill_chunk_ms) on top of a static table. An idle server
    /// keeps full chunks. Bounded requests' prefill numerics differ from a
    /// full-chunk run. 0 disables. Recipe key: decode_gap_target_ms.
    #[arg(long, default_value_t = 150, value_name = "MS")]
    pub decode_gap_target_ms: u32,

    /// Minimum free GPU memory (in MB) to keep as a safety margin during
    /// model loading. If free memory drops below this threshold after any
    /// shard, loading is aborted to prevent system OOM. Default 4096 MB
    /// accounts for CUDA context, NCCL buffers, and allocator overhead.
    #[arg(long, default_value_t = 4096)]
    pub oom_guard_mb: usize,

    // ── Parallelism ──
    /// Global rank (0=head, 1=worker, …). Only used when --world-size > 1.
    #[arg(long, default_value_t = 0)]
    pub rank: usize,

    /// Total physical ranks across all parallelism dims. Set to 2 for two-node
    /// deployment. Must satisfy `world_size == tp_size × ep_size` (orthogonal
    /// mesh) or `world_size == tp_size == ep_size` (overlapping groups on the
    /// same physical ranks — used for 2-GPU TP+EP composition).
    #[arg(long, default_value_t = 1)]
    pub world_size: usize,

    /// Tensor-parallel dimension. Splits attention/MLP weights column- and
    /// row-parallel across `tp_size` ranks. 1 = no TP. Composes with EP:
    /// MoE expert weights stay EP-sharded; attention/MLP get TP-sharded.
    #[arg(long, default_value_t = 1)]
    pub tp_size: usize,

    /// Expert-parallel dimension. Splits MoE expert weights across `ep_size`
    /// ranks. 1 = no EP. Default of 1 keeps single-rank semantics.
    #[arg(long, default_value_t = 1)]
    pub ep_size: usize,

    /// Pipeline-parallel dimension. Splits model layers into contiguous stage
    /// slices (`num_hidden_layers / pp_size`) across `pp_size` stage groups
    /// (e.g. `--pp-size 3` on 48-layer Qwen3.8-Flash-Next places 16 layers =
    /// 4 `(3 GDN + 1 QSA)` hybrid blocks per Spark). 1 = no PP.
    #[arg(long, default_value_t = 1)]
    pub pp_size: usize,

    /// Serving role for unified vs disaggregated prefill/decode topologies
    /// (`1P + 1D` / `1P + 2D`): `unified`, `prefill`, or `decode`.
    #[arg(long, default_value = "unified")]
    pub role: String,

    /// Base HTTP URL(s) of the prefill peer(s) (comma-separated, e.g.
    /// `http://10.0.0.1:8000`) when running with `--role decode` in a
    /// `1P + 1D` / `1P + 2D` disaggregated cluster.
    #[arg(long, value_name = "URL")]
    pub disagg_peer: Option<String>,

    /// NCCL bootstrap address (IP of rank 0 node).
    #[arg(long, default_value = "127.0.0.1")]
    pub master_addr: String,

    /// NCCL bootstrap port.
    #[arg(long, default_value_t = 29500)]
    pub master_port: u16,

    /// Tool call parser format. Enables OpenAI-compatible tool calling.
    /// Supported: "hermes", "qwen3_coder", "minimax_xml", "gemma4", "mistral",
    /// "glm_xml", "dsml_v41", "bare_json".
    /// When set, tool definitions in requests are injected into the system
    /// prompt and model output is parsed for tool-call envelopes.
    #[arg(long, value_name = "FORMAT")]
    pub tool_call_parser: Option<String>,

    /// Maximum output tokens per tool-calling request. Caps max_tokens from the
    /// client when tools are active to prevent unbounded generation if the model
    /// doesn't emit a </tool_call> stop token. Must be high enough for Write
    /// tool calls with large file content. Default 8192.
    #[arg(long, default_value_t = 8192)]
    pub tool_max_tokens: usize,

    /// Output allowance for a request that omits `max_tokens` /
    /// `max_completion_tokens` (chat, completions) or `max_output_tokens`
    /// (responses). Unset keeps the 4096-token wire default. A defaulted
    /// allowance is clamped to the context left after the prompt instead of
    /// rejecting the request; an explicit client value is never changed.
    #[arg(long, value_name = "TOKENS")]
    pub default_max_tokens: Option<usize>,

    /// Number of SSM state snapshot slots for Marconi prefix caching.
    /// Each slot stores SSM h_state + conv_state for all SSM layers,
    /// enabling full prefix skip (KV + SSM) on cache hits.
    /// 0 = disabled. 16 = recommended for repeated-prefix and multi-turn workloads.
    /// Intermediate checkpoints (--ssm-checkpoint-interval) require extra slots:
    /// ~(max_context / checkpoint_interval_tokens) per cached sequence.
    #[arg(long, default_value_t = 16)]
    pub ssm_cache_slots: usize,

    /// Save SSM state snapshots at regular block boundaries during prefill.
    /// When set to N > 0, a snapshot is saved every N blocks during chunked
    /// prefill. On future prefix cache hits, the deepest intermediate snapshot
    /// is restored, reducing SSM recomputation from the full prefix to just
    /// the tokens between the checkpoint and the match point.
    /// 0 = disabled (leaf-only snapshots). 256 = every 4096 tokens (block_size=16).
    #[arg(long, default_value_t = 256)]
    pub ssm_checkpoint_interval: usize,

    /// Enable automatic context compaction for long conversations.
    /// **DISABLED BY DEFAULT** (2026-04-25): the auto-compactor has
    /// historically been a source of agent loops — synthesised
    /// continuation messages and middle-of-history truncation
    /// themselves trigger drift (cf. opencode issues #15533, #17169,
    /// #19339). Oversize requests get a clean 400 error
    /// (`Prompt too long`) rather than a silently-rewritten context.
    ///
    /// Only pass `--auto-compact[=THRESHOLD]` if you have explicitly
    /// validated that compaction is safe for your model + workload.
    /// Without a value: threshold=0.75 (compact at 75% of max_seq_len).
    /// With a value: compact at that fraction (e.g., 0.80 = 80%).
    ///
    /// Method: Active Context Compression (arXiv:2601.07190) — the
    /// server uses the model itself to summarize older conversation
    /// turns into a condensed knowledge block.
    #[arg(long, value_name = "THRESHOLD", num_args = 0..=1, default_missing_value = "0.75")]
    pub auto_compact: Option<f32>,

    /// Default top-n-sigma for sampling (filter tokens by logit z-score),
    /// used when the model's generation_config.json does not set one.
    /// 0.0 = disabled (the default, as in HF/vLLM). A request's own
    /// `top_n_sigma` always wins.
    ///
    /// It was 1.0. Measured 2026-09-23 at each model's default sampling
    /// (T=1, top_k=20, top_p=0.95), 18 samples per cell: Qwen3.8-27B and
    /// Qwen3.8-Flash-Next showed 0 repetition loops with it on or off and
    /// no quality difference (top_k=20 already truncates harder), while
    /// DeepSeek-V4.1, whose config has no top_k/top_p, looped at T=1 with
    /// 1.0 as its only truncation. top-n-σ is temperature-invariant (Tang
    /// et al., arXiv:2411.07641); set 1.0 here to opt back in.
    #[arg(long, default_value_t = 0.0)]
    pub default_top_n_sigma: f32,

    /// Default min-p for sampling (keep tokens with prob >= min_p * max_prob).
    /// 0.0 = disabled. Recommended: 0.05-0.1.
    #[arg(long, default_value_t = 0.0)]
    pub default_min_p: f32,

    /// Swap space in GB for KV cache overflow to disk. When GPU blocks are
    /// exhausted, sequences are swapped to disk and resumed later.
    /// 0 = disabled. Swap files stored in /tmp/flux-swap/.
    #[arg(long, default_value_t = 3)]
    pub swap_space_gb: usize,

    /// Enable lossless block-level KV streaming via io_uring and a
    /// predictor-driven scratch pool (`spark-storage`). Coexists with
    /// `--swap-space-gb` (sequence-level evict/restore). Disabled by default;
    /// the `--high-speed-swap-*` flags below have built-in defaults if omitted.
    #[arg(long, default_value_t = false)]
    pub high_speed_swap: bool,

    /// Directory for the per-layer NVMe-backed KV files when
    /// `--high-speed-swap` is set (defaults to `/tmp/atlas-hss` if omitted).
    /// Keep on a separate path from `--swap-space-gb`'s `/tmp/flux-swap` to
    /// avoid file collisions.
    #[arg(long)]
    pub high_speed_swap_dir: Option<std::path::PathBuf>,

    /// Total disk budget for --high-speed-swap, in GiB.
    #[arg(long)]
    pub high_speed_swap_gb: Option<u64>,

    /// HBM scratch slot count (number of resident blocks).
    #[arg(long)]
    pub high_speed_swap_resident_blocks: Option<u32>,

    /// Predictor low-rank dimension (default r=32).
    #[arg(long, default_value_t = 32)]
    pub high_speed_swap_rank: u32,

    /// io_uring submission queue depth (default 32). Modern Gen5 NVMe drives
    /// saturate at QD>=32 when combined with fixed buffers and SQPOLL.
    #[arg(long, default_value_t = 32)]
    pub high_speed_swap_qd: u32,

    /// Capture the per-layer body in a CUDA graph and replay.
    /// Defaults to mirror --high-speed-swap.
    #[arg(long)]
    pub high_speed_swap_graph: Option<bool>,

    /// Per-sequence HBM cache cap for `--high-speed-swap`.
    /// When set together with --high-speed-swap, each sequence is limited
    /// to N HBM-resident KV blocks; older blocks are evicted to disk and
    /// streamed back via the orchestrator on demand. The KV cache total
    /// allocation shrinks to roughly `max_batch_size × N` blocks. Default
    /// 64 (= 1024 tokens HBM-resident at block_size=16). Set to
    /// max_seq_len/block_size to disable HBM-shrink (no eviction; useful
    /// for diff-against-no-swap correctness checks).
    #[arg(long, default_value_t = 64)]
    pub high_speed_swap_cache_blocks_per_seq: u32,

    /// Default request timeout in seconds. 0 = no timeout.
    #[arg(long, default_value_t = 300)]
    pub request_timeout: u32,

    /// Enable per-kernel profiling: sync + time each operation within layers.
    /// Disables CUDA graphs for accurate per-op timing. Adds ~10% overhead.
    #[arg(long, default_value_t = false)]
    pub profile: bool,

    /// Number of warmup tokens for online FP8 KV cache scale calibration.
    /// During the first N tokens, tracks max |K| and max |V| values across
    /// all attention layers. After N tokens, computes per-tensor scales as
    /// max/448 (mapping the observed range to FP8 E4M3 [-448, 448]).
    /// 0 = disabled (use static scales from checkpoint, or uncalibrated 1.0).
    /// Only applies when --kv-cache-dtype is fp8.
    #[arg(long, default_value_t = 0)]
    pub fp8_kv_calibration_tokens: usize,

    /// Path to a warmup prompt file (JSON messages or plain text).
    /// At startup, the server tokenizes and prefills this prompt, inserting the
    /// resulting KV cache + SSM snapshot into the prefix cache. This eliminates
    /// the cold-start TTFT penalty (~196ms) on the first real request.
    #[arg(long)]
    pub warmup_prompt: Option<std::path::PathBuf>,

    /// Enable adaptive sampling (entropy-based greedy gating, zone detection).
    /// Computes Shannon entropy over the full vocabulary per token to dynamically
    /// switch between greedy and sampled decoding. Improves quality for mixed
    /// content (code + prose) at the cost of ~2-3x decode throughput reduction.
    /// Off by default for maximum throughput.
    #[arg(long, default_value_t = false)]
    pub adaptive_sampling: bool,

    /// Disable the InstantTensor-style fast weight loader and use the mmap
    /// loader instead. The fast loader (O_DIRECT + pipelined reader/copier,
    /// with a per-shard heuristic that picks between O_DIRECT and buffered
    /// reads) is on by default — this flag is an escape hatch for rare
    /// filesystems that misbehave with O_DIRECT or for A/B debugging.
    /// Setting `ATLAS_FAST_LOAD=0` has the same effect.
    #[arg(long, default_value_t = false)]
    pub no_fast_load: bool,

    /// Address to bind the HTTP listener to. Defaults to `127.0.0.1` so a
    /// fresh install is reachable only from the local machine; pass
    /// `0.0.0.0` to expose on all interfaces (the server logs a warning
    /// when it does).
    #[arg(long, alias = "host", default_value = "127.0.0.1", value_name = "ADDR")]
    pub bind: String,

    /// Require an `Authorization: Bearer <token>` or `x-api-key: <token>` header
    /// on `/v1/*`, `/tokenize`, `/detokenize`, `/api/tags`, `/api/show`,
    /// `/flux/modes`, and `/recipes`. The token must match one loaded via
    /// `--auth-tokens-file`, `--auth-token`, or `FLUX_API_KEY`. `/`, `/ready`,
    /// `/health`, `/health/live`, and `/metrics` stay open as probe/UI targets.
    ///
    /// Defaults to off — Flux is local-by-default. Enable whenever the server is
    /// reachable from anywhere other than `localhost` (for example with
    /// `--bind 0.0.0.0` or behind a port-forward), paired with a TLS reverse proxy.
    #[arg(long, default_value_t = false)]
    pub require_auth: bool,

    /// Path to a file containing valid bearer tokens, one per line. Blank
    /// lines and lines starting with `#` are ignored. Permissions should
    /// be `0600`. The file is read once at startup; SIGHUP reloading is
    /// not supported (restart the server to rotate keys).
    #[arg(long, value_name = "PATH", conflicts_with = "auth_token")]
    pub auth_tokens_file: Option<std::path::PathBuf>,

    /// A single inline bearer token. Convenient for quick starts; not
    /// recommended for production because the token is visible in
    /// `ps`/`/proc/<pid>/cmdline`. Use `--auth-tokens-file` instead.
    #[arg(
        long,
        alias = "api-key",
        value_name = "TOKEN",
        conflicts_with = "auth_tokens_file"
    )]
    pub auth_token: Option<String>,
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::ServeArgs;

    #[test]
    fn thinking_and_auth_cli_aliases_and_reserves_parse_cleanly() {
        let parsed = ServeArgs::parse_from([
            "spark",
            "org/model",
            "--thinking-budget",
            "1536",
            "--thinking-reserve",
            "2048",
            "--thinking-off-budget",
            "256",
            "--api-key",
            "inference:sk-test",
        ]);
        assert_eq!(parsed.max_thinking_budget, Some(1536));
        assert_eq!(parsed.thinking_reserve, Some(2048));
        assert_eq!(parsed.thinking_off_budget, Some(256));
        assert_eq!(parsed.auth_token.as_deref(), Some("inference:sk-test"));
    }

    #[test]
    fn dflash_gamma_is_family_derived_unless_explicitly_overridden() {
        let automatic = ServeArgs::parse_from(["spark", "org/model", "--dflash"]);
        assert_eq!(automatic.dflash_gamma, None);

        let explicit =
            ServeArgs::parse_from(["spark", "org/model", "--dflash", "--dflash-gamma", "7"]);
        assert_eq!(explicit.dflash_gamma, Some(7));
    }

    #[test]
    fn dspark_verify_mode_default_and_explicit_value_are_preserved() {
        let automatic = ServeArgs::parse_from(["spark", "org/model", "--dflash"]);
        assert_eq!(automatic.dspark_verify_mode, "static");

        let explicit = ServeArgs::parse_from([
            "spark",
            "org/model",
            "--dflash",
            "--dspark-verify-mode",
            "compact",
        ]);
        assert_eq!(explicit.dspark_verify_mode, "compact");
    }

    #[test]
    fn dflash_draft_sample_method_defaults_to_greedy_and_preserves_explicit_value() {
        let automatic = ServeArgs::parse_from(["spark", "org/model", "--dflash"]);
        assert_eq!(automatic.dflash_draft_sample_method, "greedy");

        let explicit = ServeArgs::parse_from([
            "spark",
            "org/model",
            "--dflash",
            "--dflash-draft-sample-method",
            "probabilistic",
        ]);
        assert_eq!(explicit.dflash_draft_sample_method, "probabilistic");
    }

    #[test]
    fn display_carveout_command_follows_the_separator_and_requires_lock() {
        use super::{EntryCli, EntryCommand};
        use std::path::PathBuf;

        let cli = EntryCli::try_parse_from([
            "spark",
            "display-carveout",
            "--lock",
            "/run/x.lock",
            "--",
            "spark",
            "serve",
            "--port=1",
        ])
        .unwrap();
        let EntryCommand::DisplayCarveout(args) = cli.command else {
            panic!("expected DisplayCarveout, parsed {cli:?}");
        };
        assert_eq!(args.lock, PathBuf::from("/run/x.lock"));
        assert_eq!(args.command, ["spark", "serve", "--port=1"]);

        assert!(EntryCli::try_parse_from(["spark", "display-carveout", "--", "spark"]).is_err());
        assert!(
            EntryCli::try_parse_from(["spark", "display-carveout", "--lock", "/run/x.lock"])
                .is_err()
        );
    }
}
