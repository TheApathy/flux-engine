// SPDX-License-Identifier: AGPL-3.0-only

#![deny(clippy::all)]

//! Atlas CUDA kernel PTX modules.
//!
//! Single source of truth for embedded PTX. The `spark-runtime`
//! (pure Rust engine) and benchmarks consume these.
//!
//! PTX modules are grouped by [`KernelTarget`] — each `(H, M_q)`
//! tuple maps to a distinct set of hyperoptimized kernels.
//!
//! Constants, `ptx_modules()`, and `all_ptx_sets()` are auto-generated
//! by `build.rs` from the `kernels/{hw}/{model}/{quant}/` directories.
//! `ATLAS_TARGET_MODEL` accepts one exact model directory, a strict
//! comma-separated list of exact directories, or `*`; `ATLAS_TARGET_QUANT=*`
//! expands all quantizations. Multi-target builds are available at runtime.

use atlas_core::target::KernelTarget;

mod behavior_defaults;
pub use behavior_defaults::DEFAULT_MAX_INTER_TOOL_PROSE;

mod drafter_profile;
pub use drafter_profile::{
    DrafterAdmission, DrafterCaptureMode, DrafterIdentity, DrafterKind, DrafterProfile,
    TargetIdentity, VocabPin, drafter_profiles, drafter_profiles_for,
};

// Auto-generated: per-target PTX constants, ptx_modules() function,
// and all_ptx_sets() for multi-target builds.
include!(concat!(env!("OUT_DIR"), "/target_ptx.rs"));

// ═══════════════════════════════════════════════════════════════════
// Target-aware PTX grouping
// ═══════════════════════════════════════════════════════════════════

/// Per-category sampling defaults from MODEL.toml.
#[derive(Debug, Clone, Copy)]
pub struct SamplingCategory {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// Multiplicative penalty on already-seen tokens (1.0 = disabled).
    /// Populated from MODEL.toml `[sampling.*].repetition_penalty` via build.rs.
    pub repetition_penalty: f32,
    /// DRY (Don't-Repeat-Yourself) sampler parameters. Penalises tokens
    /// that extend repeated n-grams past `dry_allowed_length` with an
    /// exponential `dry_multiplier * dry_base^(match_len - allowed)` —
    /// the targeted fix for phrase-level attractors (e.g. the
    /// ```` ```bash cd … cargo test ``` ```` fence-narration loop
    /// observed in Qwen3.5-35B-A3B-FP8 opencode sessions at turn ≥ 8).
    ///
    /// `presence_penalty` on its own is a FLAT per-unique-token hit
    /// (does not scale with repetition count), so it can't break a
    /// phrase attractor where individual tokens already paid their
    /// penalty once. DRY scales with the repeat-length and is the
    /// published remedy (oobabooga/text-generation-webui#5677, used in
    /// llama.cpp / Aphrodite / TabbyAPI).
    ///
    /// `dry_multiplier = 0.0` disables DRY for this category (default
    /// for every preset unless MODEL.toml sets it explicitly).
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    /// LZ penalty (arXiv:2504.20131). Per-extension n-gram penalty
    /// over a 256-token rolling window. Frequency-weighted and length-
    /// scaled, so it correctly distinguishes "phrase loop" from
    /// "legitimate vocabulary reuse" without the flat-per-token
    /// `presence_penalty` regression. 0.0 = disabled. SGLang reference
    /// strength = 0.2 (lossless on AIME/GPQA).
    pub lz_penalty: f32,
    /// Model-declared min-p. `None` preserves the server CLI default;
    /// `Some(0.0)` deliberately disables min-p for this model/category.
    pub min_p: Option<f32>,
    /// Model-declared top-n-sigma. `None` preserves the server CLI default;
    /// `Some(0.0)` deliberately disables the filter for this category.
    pub top_n_sigma: Option<f32>,
}

/// Model-specific sampling presets loaded from MODEL.toml `[sampling.*]`.
#[derive(Debug, Clone, Copy)]
pub struct SamplingPresets {
    pub thinking_text: SamplingCategory,
    pub thinking_coding: SamplingCategory,
    pub non_thinking: SamplingCategory,
    /// Tool-calling preset: model-recommended sampling for agentic tasks.
    /// Qwen3.5 recommends temperature=0.6 (NOT greedy) to avoid repetition loops.
    pub tools: SamplingCategory,
}

impl Default for SamplingPresets {
    fn default() -> Self {
        let default_cat = SamplingCategory {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            // DRY defaults = disabled (multiplier 0.0). Per-MODEL.toml
            // tools presets opt in when the model needs it.
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            lz_penalty: 0.0,
            min_p: None,
            top_n_sigma: None,
        };
        let tools_cat = SamplingCategory {
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            lz_penalty: 0.0,
            min_p: None,
            top_n_sigma: None,
        };
        Self {
            thinking_text: default_cat,
            thinking_coding: default_cat,
            non_thinking: default_cat,
            tools: tools_cat,
        }
    }
}

/// Model-specific behavior flags from MODEL.toml `[behavior]`.
#[derive(Debug, Clone)]
pub struct ModelBehavior {
    /// Allow thinking when tools are active. Default: true.
    pub thinking_in_tools: bool,
    /// Maximum thinking budget (tokens). Default: 262144.
    pub max_thinking_budget: u32,
    /// Default thinking state for this model when the client request does not
    /// specify a reasoning_effort / thinking parameter. Typical values:
    /// - thinking-first models (Mistral Small 4, Qwen3.5, …): `true`
    /// - instruct-only models with no `<think>` tokens: `false`
    ///
    /// Overridden per-request by `reasoning_effort`, and globally by the
    /// `--disable-thinking` CLI flag.
    pub thinking_default: bool,
    /// When an EOS token is sampled while the model is still inside a
    /// `<think>` block, implicitly close the thinking phase and discard that
    /// first EOS. The next sampled token then gets one content-phase chance;
    /// a second EOS terminates normally. Default: false because this changes
    /// model generation semantics and must be qualified per target.
    pub honor_eos_inside_thinking: bool,
    /// Default FP8 KV calibration tokens (0 = disabled).
    pub fp8_kv_calibration_tokens: usize,
    /// Default KV cache dtype from MODEL.toml (e.g., "bf16", "fp8").
    /// When non-empty, overrides the CLI default for models that need
    /// higher precision. User can still override with explicit --kv-cache-dtype.
    pub default_kv_dtype: &'static str,
    /// Default num_drafts for speculative decoding (0 = use CLI default).
    /// K = num_drafts + 1 (num_drafts=1 → K=2 verifies 2 tokens per step).
    /// Optimal K varies per model; benchmarks sometimes show K=2 beats K=3.
    /// User override with --num-drafts still wins.
    pub default_num_drafts: u32,
    /// Skip the `<tool_call>\n` steering prefix in the chat template's
    /// generation prompt. Some Nemotron variants (Super 120B) weren't
    /// trained on qwen3_coder XML and emit a `<tool_call>` token loop
    /// when the prefix forces them into that structure. Default: false
    /// (keep the existing Nemotron-Nano-correct behavior).
    pub disable_tool_steering: bool,
    /// Per-model tool-call parser override. Empty string = use the
    /// `tool_defaults.toml` mapping for this `model_type`. Set in MODEL.toml
    /// `[behavior].tool_call_parser` when one variant of a model_type needs
    /// a different parser than its siblings (e.g. Nemotron-Super-120B uses
    /// `bare_json` while Nemotron-Nano-30B stays on `qwen3_coder`).
    pub tool_call_parser: &'static str,
    /// Enable the content-loop watchdog (period-N token-repetition detector
    /// at `decode_logits_step.rs:230`). Default: `false` — most models
    /// terminate cleanly via EOS / `max_tokens` without it. Models with a
    /// known prose-attractor failure mode (Qwen3.5-35B-A3B's "Running:```bash
    /// cmd```Executing:" loop, observed during agentic Claude Code sessions)
    /// should set this `true` in MODEL.toml `[behavior]`.
    ///
    /// The watchdog has false-positives on legitimate structured output
    /// (chess board JS init `{color:BLACK,type:'P'},` × 8, HTML tables,
    /// JSON arrays of similar objects, multiplication tables). Enable only
    /// when the model has been observed to need it.
    pub enable_loop_watchdog: bool,
    /// Thinking-loop watchdog: substring-occurrence count that trips a
    /// forced `</think>`. Default 3 (historical `THINK_LOOP_MIN_REPEATS`).
    pub think_loop_min_repeats: u32,
    /// Thinking-loop watchdog: trailing-token scan window. Default 160.
    pub think_loop_scan_window: u32,
    /// Confidence-run early-stop enabled. Default `false`. Set `true`
    /// only for models qualified for this heuristic (keep `false` for
    /// models whose deterministic code drafting trips it).
    pub confidence_early_stop: bool,
    /// Confidence run length before arming forced `</think>`.
    /// Default 30.
    pub confidence_run_length: u32,
    /// Fuzzy-repetition detector Hamming tolerance divisor: a
    /// `pattern_len`-token window tolerates `pattern_len / div`
    /// mismatches. Default 12 (~8%).
    pub fuzzy_repeat_tolerance_div: u32,
    /// Cap on free-text tokens between successive `<tool_call>` opens in
    /// `tool_choice=auto`. Default [`DEFAULT_MAX_INTER_TOOL_PROSE`].
    pub max_inter_tool_prose: u32,
    /// TSCG (Tool-Schema Compilation) enabled — compile tool JSON
    /// schemas to compact function signatures before prompting.
    /// Default `false`; the TAS operator is tokenizer-specific so
    /// enable + verify per model. arXiv:2605.04107.
    pub tscg: bool,
    /// Disable XGrammar tool-call constrained decoding for this model.
    /// Default `false`. Escape hatch for the "structure snowballing"
    /// alignment tax (arXiv:2604.06066) — a few models tool-call more
    /// reliably unconstrained. When `true`, tool calls are parsed but
    /// not grammar-enforced.
    pub disable_tool_grammar: bool,
    /// When a decode-time watchdog (content-loop, fuzzy-repeat,
    /// inter-tool prose) detects degeneration, roll the sequence back to
    /// the last well-formed boundary and let generation re-steer, instead
    /// of hard-stopping the response. Default `true` (recovers responses,
    /// especially mid-tool-call — arXiv:2603.27905 ATLAS-RTC). Set `false`
    /// to keep the legacy hard-stop behavior. Capped at
    /// [`crate::ROLLBACK_RESTEER_CAP`] rollbacks per sequence, after which
    /// the hard-stop fires regardless.
    pub rollback_resteer: bool,
    /// ROM (arXiv:2603.22016) scaffold. Path to a trained
    /// repetition-onset detection head artifact. Empty string = no ROM
    /// head; the confidence-run heuristic stays as the fallback. A trained
    /// artifact can be dropped in later via MODEL.toml
    /// `[behavior].rom_head` without further code changes — the runtime
    /// loads it through the `RomHead` trait seam. The detector
    /// itself is intentionally NOT implemented (no per-model trained head
    /// is available); only the optional hook is wired.
    pub rom_head: &'static str,
    /// Optional Jinja `preserve_thinking` pin. `None` leaves the variable
    /// undefined so each checkpoint template keeps its native default.
    pub preserve_thinking: Option<bool>,
    /// Answer room (tokens) kept out of a thinking allowance the client did
    /// not size. When non-zero, an omitted-budget request may think for at
    /// most `max_tokens - max(15% of max_tokens, min(reserve, max_tokens / 2))`
    /// in addition to the existing 90% (70% with tools) cap. Default 0 keeps
    /// the existing cap alone.
    pub thinking_answer_reserve: u32,
    /// Tool turns with thinking enabled sample with the `thinking_text`
    /// preset instead of `tools`. Default false keeps the `tools` preset.
    pub thinking_sampling_in_tools: bool,
}

/// Maximum number of watchdog-triggered rollbacks a single
/// sequence may perform before the watchdog reverts to a hard stop.
/// Bounds the worst case where re-steering re-enters the same attractor
/// — without this a degenerate sequence could rollback indefinitely.
pub const ROLLBACK_RESTEER_CAP: u32 = 2;

impl Default for ModelBehavior {
    fn default() -> Self {
        Self {
            thinking_in_tools: true,
            max_thinking_budget: 262144,
            thinking_default: false,
            honor_eos_inside_thinking: false,
            fp8_kv_calibration_tokens: 0,
            default_kv_dtype: "",
            default_num_drafts: 0,
            disable_tool_steering: false,
            tool_call_parser: "",
            enable_loop_watchdog: false,
            think_loop_min_repeats: 3,
            think_loop_scan_window: 160,
            confidence_early_stop: false,
            confidence_run_length: 30,
            fuzzy_repeat_tolerance_div: 12,
            max_inter_tool_prose: DEFAULT_MAX_INTER_TOOL_PROSE,
            tscg: false,
            disable_tool_grammar: false,
            rollback_resteer: true,
            rom_head: "",
            preserve_thinking: None,
            thinking_answer_reserve: 0,
            thinking_sampling_in_tools: false,
        }
    }
}

/// Declares which `(model_type, hidden_size)` pairs a kernel target supports.
/// Parsed from `[[model_types]]` in MODEL.toml at build time.
pub struct ModelTypeMatch {
    pub model_type: &'static str,
    /// `None` = wildcard (matches any hidden_size not caught by a more specific entry).
    pub hidden_size: Option<usize>,
}

/// DFlash speculative-decoding pairing for a target model.
/// Parsed from `[dflash]` in MODEL.toml at build time. `None` when the
/// model has no DFlash drafter associated.
#[derive(Debug, Clone)]
pub struct DflashConfig {
    /// HuggingFace id (or local path) of the drafter checkpoint.
    pub draft_model: &'static str,
    /// Block size γ (parallel draft tokens per step). Defaults to 16.
    pub gamma: usize,
    /// Drafter sliding-window size in tokens. 0 = full attention.
    pub window_size: usize,
    /// Token id used to fill the γ "to-be-predicted" positions during
    /// drafter forward. From the drafter's `dflash_config.mask_token_id`.
    pub mask_token_id: u32,
    /// Target-side layer indices to capture intermediate hidden states from
    /// (shallow-to-deep). The drafter's `fc` projection consumes the stack
    /// of these hiddens. From the drafter's `dflash_config.target_layer_ids`.
    pub target_layer_ids: &'static [usize],
}

/// PTX modules hyperoptimized for a specific (H, M_q) target.
pub struct TargetPtxSet {
    pub target: KernelTarget,
    pub modules: Vec<(&'static str, &'static str)>,
    pub sampling: SamplingPresets,
    pub behavior: ModelBehavior,
    pub model_type_matches: Vec<ModelTypeMatch>,
    /// DFlash drafter pairing for this model. `None` when the MODEL.toml has
    /// no `[dflash]` section. Consumed by spark-server when `--dflash` is
    /// set without an explicit `--draft-model` flag.
    pub dflash: Option<DflashConfig>,
}

/// All compiled kernel targets and their PTX module sets.
///
/// Returns one entry per target compiled at build time.
/// Single-target builds return one entry; wildcard builds return all.
pub fn available_targets() -> Vec<TargetPtxSet> {
    register_embedded_aot_cubins();
    all_ptx_sets()
}

/// Find the PTX module set for a target whose model name contains `needle`.
///
/// Returns `None` if no compiled target matches.
pub fn ptx_for_model(needle: &str) -> Option<TargetPtxSet> {
    register_embedded_aot_cubins();
    all_ptx_sets()
        .into_iter()
        .find(|t| t.target.model.contains(needle))
}

/// Find the PTX module set matching a `(model_type, hidden_size)` pair.
///
/// Matching rules:
/// 1. Exact match on `(model_type, Some(hidden_size))` wins
/// 2. Wildcard match `(model_type, None)` is fallback
/// 3. Returns `None` if no compiled target matches
pub fn ptx_for_config(model_type: &str, hidden_size: usize) -> Option<TargetPtxSet> {
    register_embedded_aot_cubins();
    let targets = all_ptx_sets();
    // Exact match first (specific hidden_size)
    let exact = targets.iter().position(|t| {
        t.model_type_matches
            .iter()
            .any(|m| m.model_type == model_type && m.hidden_size == Some(hidden_size))
    });
    if let Some(idx) = exact {
        return targets.into_iter().nth(idx);
    }
    // Wildcard fallback (hidden_size == None)
    let wildcard = targets.iter().position(|t| {
        t.model_type_matches
            .iter()
            .any(|m| m.model_type == model_type && m.hidden_size.is_none())
    });
    wildcard.and_then(|idx| targets.into_iter().nth(idx))
}

#[cfg(test)]
#[path = "../build_codegen.rs"]
mod build_codegen_under_test;

#[cfg(test)]
#[path = "../build_hw.rs"]
mod build_hw_under_test;

#[cfg(test)]
#[path = "../build_deps.rs"]
#[expect(
    dead_code,
    reason = "remaining build_deps helpers are tested in tests/build_deps.rs"
)]
mod build_deps_under_test;

#[cfg(test)]
#[path = "../build_profile.rs"]
mod build_profile_under_test;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_ptx_modules_non_empty() {
        for (name, ptx) in ptx_modules() {
            assert!(
                !ptx.is_empty(),
                "PTX module '{name}' is empty — nvcc compilation may have failed"
            );
            assert!(
                ptx.contains(".version"),
                "PTX module '{name}' doesn't contain .version directive"
            );
        }
    }

    // These tests assert that PTX modules were actually compiled into the
    // crate at build time. They require nvcc + a real CUDA toolchain — the
    // CI host runs with `ATLAS_SKIP_BUILD=1`, which emits an empty stub
    // registry by design (so `cargo check` / `cargo clippy` / `cargo test`
    // can run on hosts without a GPU). Mark them `#[ignore]` so default
    // `cargo test` is green; they're still exercised on a developer
    // machine via `cargo test -p atlas-kernels -- --ignored` after a
    // real PTX build.

    #[test]
    #[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
    fn module_count_matches_cu_files() {
        let count = ptx_modules().len();
        assert!(count >= 31, "Expected at least 31 PTX modules, got {count}");
    }

    #[test]
    #[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
    fn available_targets_non_empty() {
        let targets = available_targets();
        assert!(!targets.is_empty(), "No kernel targets available");
        assert!(
            targets.iter().any(|t| t.target.quant == "nvfp4"),
            "Expected at least one NVFP4 target"
        );
    }

    #[test]
    #[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
    fn all_targets_have_modules() {
        for t in available_targets() {
            assert!(
                t.modules.len() >= 31,
                "Target {} has only {} modules (expected >= 31)",
                t.target,
                t.modules.len()
            );
        }
    }

    #[test]
    #[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
    fn ptx_for_model_lookup() {
        let found = ptx_for_model("qwen3.8-flash-next");
        assert!(
            found.is_some(),
            "ptx_for_model('qwen3.8-flash-next') should find the default target"
        );
    }

    #[test]
    fn behavior_default_prose_budget_uses_shared_plan_sized_constant() {
        let behavior = ModelBehavior::default();
        assert_eq!(behavior.max_inter_tool_prose, DEFAULT_MAX_INTER_TOOL_PROSE);
        assert!(behavior.max_inter_tool_prose >= 2048);
        assert_eq!(behavior.preserve_thinking, None);
        assert!(!behavior.honor_eos_inside_thinking);
        assert_eq!(behavior.thinking_answer_reserve, 0);
        assert!(!behavior.thinking_sampling_in_tools);
    }

    #[test]
    fn should_embed_cubin_policy() {
        use build_codegen_under_test::should_embed_cubin;

        // Default ON for single-target concrete SM CUDA builds.
        assert!(should_embed_cubin(true, 1, "sm_121f", None));
        assert!(should_embed_cubin(true, 1, "sm_121a", None));
        assert!(should_embed_cubin(true, 1, "sm_120a", None));
        assert!(should_embed_cubin(true, 1, "sm_120f", None));
        assert!(should_embed_cubin(true, 1, "sm_103a", None));
        assert!(should_embed_cubin(true, 1, "sm_90a", None));

        // Default OFF for multi-target universal builds (keeps universal lean).
        assert!(!should_embed_cubin(true, 2, "sm_121f", None));
        assert!(!should_embed_cubin(true, 14, "sm_121f", None));
        assert!(!should_embed_cubin(true, 0, "sm_121f", None));

        // Explicit env overrides win for valid SM CUDA builds.
        assert!(should_embed_cubin(true, 14, "sm_121f", Some("1")));
        assert!(should_embed_cubin(true, 14, "sm_121f", Some("true")));
        assert!(!should_embed_cubin(true, 1, "sm_121f", Some("0")));
        assert!(!should_embed_cubin(true, 1, "sm_121f", Some("false")));

        // Non-text backends (Metal) and non-SM virtual arches never embed CUBIN.
        assert!(!should_embed_cubin(false, 1, "sm_121f", None));
        assert!(!should_embed_cubin(false, 1, "sm_121f", Some("1")));
        assert!(!should_embed_cubin(true, 1, "compute_121", None));
        assert!(!should_embed_cubin(true, 1, "compute_121", Some("1")));
        assert!(!should_embed_cubin(true, 1, "metal3.1", Some("1")));
    }

    #[test]
    fn extract_ptxas_flags_forwards_only_ptxas_options() {
        use build_codegen_under_test::extract_ptxas_flags;

        let to_vec = |xs: &[&str]| -> Vec<String> { xs.iter().map(|s| (*s).to_owned()).collect() };

        // Standard KERNEL.toml / EXL3 nvcc flags are baked into PTX, not ptxas.
        assert!(
            extract_ptxas_flags(&to_vec(&[
                "--atlas-ieee-math",
                "--fmad=false",
                "-DHDIM=128",
                "-DTQ_PLUS_SIGNS",
                "-std=c++17",
                "-I/opt/ext",
                "-Xcudafe",
                "--diag_suppress=20012",
            ]))
            .is_empty()
        );

        // Both space-separated and `=` forms, plus comma-separated lists.
        assert_eq!(
            extract_ptxas_flags(&to_vec(&[
                "--fmad=false",
                "-Xptxas",
                "-v,--warn-on-spills",
                "-Xptxas=-dlcm=cg,-maxrregcount=128",
                "--ptxas-options",
                "-O2",
                "--ptxas-options=-dscm=wb",
                "-Xcompiler",
                "-Xptxas=-ignored",
            ])),
            to_vec(&[
                "-v",
                "--warn-on-spills",
                "-dlcm=cg",
                "-maxrregcount=128",
                "-O2",
                "-dscm=wb",
            ])
        );

        // Trailing forwarder without operand does not panic.
        assert!(extract_ptxas_flags(&to_vec(&["-Xptxas"])).is_empty());
    }

    #[test]
    fn parse_ptx_target_arch_reads_header_directive() {
        use build_codegen_under_test::parse_ptx_target_arch;

        let ptx_f = b"// Generated by NVIDIA NVVM Compiler\n.version 8.8\n.target sm_121f\n.address_size 64\n";
        assert_eq!(parse_ptx_target_arch(ptx_f).as_deref(), Some("sm_121f"));

        let ptx_a = b".version 8.8\n.target sm_121a, texmode_independent\n.address_size 64\n";
        assert_eq!(parse_ptx_target_arch(ptx_a).as_deref(), Some("sm_121a"));

        let ptx_tab = b".version 8.5\n  .target\tsm_90a\n";
        assert_eq!(parse_ptx_target_arch(ptx_tab).as_deref(), Some("sm_90a"));

        let ptx_sm120a = b".version 8.8\n.target sm_120a\n.address_size 64\n";
        assert_eq!(
            parse_ptx_target_arch(ptx_sm120a).as_deref(),
            Some("sm_120a")
        );

        let ptx_sm120f = b".version 8.8\n.target sm_120f, texmode_independent\n";
        assert_eq!(
            parse_ptx_target_arch(ptx_sm120f).as_deref(),
            Some("sm_120f")
        );

        let ptx_sm103a = b".version 8.8\n.target sm_103a\n.address_size 64\n";
        assert_eq!(
            parse_ptx_target_arch(ptx_sm103a).as_deref(),
            Some("sm_103a")
        );

        // Non-SM or malformed directives return None so build.rs falls back to HARDWARE.toml.
        assert_eq!(parse_ptx_target_arch(b".target_foo sm_121a\n"), None);
        assert_eq!(parse_ptx_target_arch(b".target compute_90\n"), None);
        assert_eq!(parse_ptx_target_arch(b".version 8.8\n"), None);
    }

    #[test]
    fn build_obfuscation_round_trips_through_atlas_core_cubin_store() {
        use atlas_core::target::{
            AotCubinEncoding, decode_obf_bytes, lookup_aot_cubin, register_aot_cubin, sha256_bytes,
        };
        use build_codegen_under_test::obfuscate_ptx_bytes;

        const PTX: &str = ".version 8.8\n.target sm_121f\n.address_size 64\n";
        const ELF: &[u8] = &[0x7f, b'E', b'L', b'F', 0x00, 0x02, 0x00, 0x01, 0x00, 0xfe];

        let obf = obfuscate_ptx_bytes(ELF);
        assert_ne!(obf.as_slice(), ELF);
        assert_eq!(decode_obf_bytes(&obf).as_slice(), ELF);

        let obf_static: &'static [u8] = Box::leak(obf.into_boxed_slice());
        register_aot_cubin(
            "__test_kernels_obf_roundtrip",
            PTX.len(),
            sha256_bytes(PTX.as_bytes()),
            obf_static,
            AotCubinEncoding::Obfuscated,
        );
        let decoded = lookup_aot_cubin("__test_kernels_obf_roundtrip", PTX)
            .expect("registered obfuscated CUBIN must decode on lookup");
        assert_eq!(decoded.as_ref(), ELF);
    }

    #[test]
    fn hw_aliases_and_env_precedence() {
        use build_hw_under_test::{canonical_hw_name, resolve_hw_env};

        assert_eq!(canonical_hw_name("sm120"), "rtx6000");
        assert_eq!(canonical_hw_name("SM120"), "rtx6000");
        assert_eq!(canonical_hw_name(" sm121 "), "gb10");
        assert_eq!(canonical_hw_name("sm103"), "gb300");
        assert_eq!(canonical_hw_name("rtx6000"), "rtx6000");
        assert_eq!(canonical_hw_name("gb10"), "gb10");

        assert_eq!(resolve_hw_env(None, None), "gb10");
        assert_eq!(resolve_hw_env(Some(""), Some("sm120")), "rtx6000");
        assert_eq!(resolve_hw_env(Some("sm103"), Some("sm120")), "gb300");
        assert_eq!(resolve_hw_env(None, Some("rtx6000")), "rtx6000");
    }

    #[test]
    fn hw_extends_inheritance_overrides_and_include_order() {
        use build_deps_under_test::include_closure_with_dirs;
        use build_hw_under_test::{
            collect_cu_files_layered, collect_include_dirs, resolve_hw_chain,
            resolve_kernel_toml_dir, resolve_model_meta_dir,
        };
        use std::path::PathBuf;

        let root = std::env::temp_dir().join(format!(
            "atlas-build-hw-inherit-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let base = root.join("gb10");
        let child = root.join("rtx6000");
        std::fs::create_dir_all(base.join("common")).unwrap();
        std::fs::create_dir_all(base.join("m/nvfp4")).unwrap();
        std::fs::create_dir_all(child.join("common")).unwrap();
        std::fs::create_dir_all(child.join("m/nvfp4")).unwrap();

        std::fs::write(
            base.join("HARDWARE.toml"),
            "[hardware]\nname = \"gb10\"\nvendor = \"nvidia\"\narch = \"sm_121f\"\nuse_fp32_residual = true\n",
        )
        .unwrap();
        std::fs::write(
            child.join("HARDWARE.toml"),
            "[hardware]\nname = \"rtx6000\"\narch = \"sm_120a\"\nextends = \"sm121\"\n",
        )
        .unwrap();
        std::fs::write(base.join("m/MODEL.toml"), "[model]\nname = \"base\"\n").unwrap();
        std::fs::write(base.join("m/nvfp4/KERNEL.toml"), "[build]\n").unwrap();

        // Kernel shadowing:
        // - base/common/a.cu inherited
        // - base/common/b.cu overridden by child/common/b.cu
        // - base/common/c.cu overridden by base/m/nvfp4/c.cu (model beats common)
        // - base/m/nvfp4/d.cu overridden by child/m/nvfp4/d.cu (child model beats base model)
        std::fs::write(base.join("common/a.cu"), "__global__ void a() {}\n").unwrap();
        std::fs::write(base.join("common/b.cu"), "__global__ void b_base() {}\n").unwrap();
        std::fs::write(child.join("common/b.cu"), "#include \"shared.cuh\"\n").unwrap();
        std::fs::write(base.join("common/c.cu"), "__global__ void c_common() {}\n").unwrap();
        std::fs::write(base.join("m/nvfp4/c.cu"), "__global__ void c_model() {}\n").unwrap();
        std::fs::write(base.join("m/nvfp4/d.cu"), "__global__ void d_base() {}\n").unwrap();
        std::fs::write(child.join("m/nvfp4/d.cu"), "#include \"base_only.cuh\"\n").unwrap();
        std::fs::write(base.join("common/shared.cuh"), "// base\n").unwrap();
        std::fs::write(child.join("common/shared.cuh"), "// child\n").unwrap();
        std::fs::write(base.join("m/nvfp4/base_only.cuh"), "// base only\n").unwrap();

        let chain = resolve_hw_chain(&root, "sm120").expect("resolve child chain");
        assert_eq!(chain.hw, "rtx6000");
        assert_eq!(chain.arch, "sm_120a");
        assert_eq!(chain.vendor.as_deref(), Some("nvidia"));
        assert_eq!(chain.use_fp32_residual, Some(true));
        assert!(chain.is_inherited());
        assert!(chain.includes_hw("gb10"));
        assert_eq!(chain.hw_dirs, vec![child.clone(), base.clone()]);

        assert_eq!(
            resolve_model_meta_dir(&chain.hw_dirs, "m"),
            Some(base.join("m"))
        );
        assert_eq!(
            resolve_kernel_toml_dir(&chain.hw_dirs, "m", "nvfp4"),
            Some(base.join("m/nvfp4"))
        );

        let cus = collect_cu_files_layered(&chain.hw_dirs, "m", "nvfp4");
        let expected: Vec<PathBuf> = {
            let mut v = vec![
                base.join("common/a.cu"),
                child.join("common/b.cu"),
                base.join("m/nvfp4/c.cu"),
                child.join("m/nvfp4/d.cu"),
            ];
            v.sort();
            v
        };
        assert_eq!(cus, expected);

        // Child include dirs precede base include dirs.
        let incs = collect_include_dirs(&chain.hw_dirs, "m", "nvfp4");
        let child_pos = incs
            .iter()
            .position(|p| p == &child.join("common"))
            .unwrap();
        let base_pos = incs.iter().position(|p| p == &base.join("common")).unwrap();
        assert!(child_pos < base_pos, "child -I must precede base -I");

        // Child .cu resolves base header via -I fallback.
        let deps = include_closure_with_dirs(&child.join("m/nvfp4/d.cu"), &incs);
        assert_eq!(
            deps,
            vec![base.join("m/nvfp4/base_only.cuh").canonicalize().unwrap()]
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn hw_extends_rejects_cycles() {
        use build_hw_under_test::resolve_hw_chain;

        let root =
            std::env::temp_dir().join(format!("atlas-build-hw-cycle-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::create_dir_all(root.join("b")).unwrap();
        std::fs::write(
            root.join("a/HARDWARE.toml"),
            "[hardware]\narch = \"sm_120a\"\nextends = \"b\"\n",
        )
        .unwrap();
        std::fs::write(
            root.join("b/HARDWARE.toml"),
            "[hardware]\narch = \"sm_121f\"\nextends = \"a\"\n",
        )
        .unwrap();

        let err = resolve_hw_chain(&root, "a").expect_err("cycle must be rejected");
        assert!(err.contains("circular"), "unexpected error: {err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn shipped_rtx6000_inherits_gb10_models_and_drafter_profiles() {
        use build_hw_under_test::{
            collect_cu_files_layered, resolve_hw_chain, resolve_kernel_toml_dir,
            resolve_model_meta_dir,
        };
        use build_profile_under_test::{
            generate_drafter_profiles_rs, generate_drafter_profiles_rs_layered,
        };
        use std::path::Path;

        let kernels_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
        let chain = resolve_hw_chain(&kernels_root, "sm120").expect("rtx6000 chain");
        assert_eq!(chain.hw, "rtx6000");
        assert_eq!(chain.arch, "sm_120a");
        assert_eq!(chain.vendor.as_deref(), Some("nvidia"));
        assert_eq!(chain.hw_names, vec!["rtx6000", "gb10"]);

        for model in ["qwen3.8-flash-next", "qwen3.8-27b"] {
            assert!(
                resolve_model_meta_dir(&chain.hw_dirs, model).is_some(),
                "{model} MODEL.toml must inherit from gb10"
            );
            assert!(
                resolve_kernel_toml_dir(&chain.hw_dirs, model, "nvfp4").is_some(),
                "{model} KERNEL.toml must inherit from gb10"
            );
            let cus = collect_cu_files_layered(&chain.hw_dirs, model, "nvfp4");
            assert!(
                cus.len() >= 100,
                "{model} should inherit gb10 kernels, got {}",
                cus.len()
            );
        }

        // Layered drafter profiles for rtx6000 must match gb10 verbatim while rtx6000 has no overrides.
        let gb10_profiles = generate_drafter_profiles_rs(&kernels_root.join("gb10"));
        let rtx_profiles = generate_drafter_profiles_rs_layered(&chain.hw_dirs);
        assert_eq!(rtx_profiles, gb10_profiles);
    }
}
