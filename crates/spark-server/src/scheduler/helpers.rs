// SPDX-License-Identifier: AGPL-3.0-only

//! Helpers: BF16 conversion, hard-stop registry, loop detection, sampling defaults.

/// Convert two little-endian BF16 bytes to f32.
#[inline]
pub fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16)
}

/// Cached `ATLAS_DUMP_HIDDEN` env var read.
///
/// Returns `Some(path)` if the env var is set (non-empty), `None` otherwise.
/// The lookup happens exactly once at first call and is cached for the
/// process lifetime. `emit_token` and `process_decode_logits` previously
/// re-read the env var on every emitted token; on a 256-token response that
/// was 256 syscalls + cstring conversions adding ~1.3 ms of pure overhead
/// when the var was unset (the production default).
#[inline]
pub fn dump_hidden_path() -> Option<&'static str> {
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        std::env::var("ATLAS_DUMP_HIDDEN")
            .ok()
            .filter(|s| !s.is_empty())
    })
    .as_deref()
}

/// Global hard-stop token for ChatML role boundaries (`<|im_start|>`).
///
/// Set once at startup from `serve_phases::tokenizer_runtime::resolve_tokenizer_runtime`
/// when the tokenizer exposes `<|im_start|>` as a single token id (Qwen3.5/3.6 family
/// tokenizers: id 248045). Read from `emit_token` to bail out of the turn
/// regardless of grammar / tool-call / min_tokens suppression — otherwise
/// the model can sample `<|im_start|>`, have it silently swallowed as a
/// suppressed EOS, and continue emitting the following role literal
/// (`user` / `assistant`, plain BPE tokens) which DO stream to the client.
///
/// 0 = unset / no hard-stop (non-Qwen tokenizers). The value is checked
/// with `load(Ordering::Relaxed)` on the emit path — no atomicity contract
/// beyond "set before the first request lands", which is guaranteed by the
/// tokenizer runtime init ordering.
static IM_START_HARD_STOP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Install or clear the ChatML role-boundary hard-stop. Called from
/// `serve_phases::tokenizer_runtime::resolve_tokenizer_runtime` at model load
/// when `<|im_start|>` resolves to a single token id; passing `0` clears any
/// previously installed hard-stop token.
pub fn set_im_start_hard_stop(id: u32) {
    IM_START_HARD_STOP.store(id, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
pub fn im_start_hard_stop() -> Option<u32> {
    let id = IM_START_HARD_STOP.load(std::sync::atomic::Ordering::Relaxed);
    if id == 0 { None } else { Some(id) }
}
// ── Sampling defaults (SSOT) ────────────────────────────────────────────────
// All SamplingParams constructors reference these constants. Change here, not
// at each call site.
pub const DEFAULT_LZ_PENALTY: f32 = 0.0;
pub const DEFAULT_DRY_MULTIPLIER: f32 = 0.0;
pub const DEFAULT_DRY_BASE: f32 = 1.75;
// `allowed_length=3` avoids penalising legitimate code micro-repetition
// (consecutive `(`, `,`, indentation, short `let x =` lines) and tool-call JSON
// emission (which `allowed_length=2` disrupts), while still catching longer
// bash-fence narration loops (6+ tokens).
pub const DEFAULT_DRY_ALLOWED_LENGTH: u32 = 3;

/// Token-level thinking-loop detection parameters. Tuned to catch
/// fence-narration attractors (e.g., repeated `Running: ... Executing: ...`
/// cycles inside `<think>`) without false-positiving on legitimate
/// numbered-list reasoning.
///
/// Strategy: once a sequence has spent `THINK_LOOP_MIN_TOKENS` inside
/// `<think>`, every `THINK_LOOP_CHECK_STRIDE` thinking tokens scan
/// the tail for a pattern of length `p ∈ [THINK_LOOP_PERIOD_MIN,
/// THINK_LOOP_PERIOD_MAX]` that repeats `THINK_LOOP_MIN_REPEATS`
/// non-overlapping times within `THINK_LOOP_SCAN_WINDOW` (allowing variable
/// text between repeats). If detected, set `force_end_thinking=true` so
/// the existing machinery force-emits `</think>` — the session
/// regains its full content budget instead of burning the thinking
/// cap. No workaround: attacks the phrase-loop attractor at its
/// earliest visible point, before it can monopolise the turn.
pub const THINK_LOOP_MIN_TOKENS: u32 = 48;
pub const THINK_LOOP_CHECK_STRIDE: u32 = 8;
pub const THINK_LOOP_PERIOD_MIN: usize = 4;
pub const THINK_LOOP_PERIOD_MAX: usize = 20;
pub const THINK_LOOP_MIN_REPEATS: usize = 3;
/// How many tokens back from the current tail to scan for needle
/// occurrences. Large enough to contain 3+ copies of a period-20
/// block (60 tokens) plus comfortable slack for the connective
/// prefixes that separate them.
pub const THINK_LOOP_SCAN_WINDOW: usize = 160;

/// A ChatML role opener is never valid reasoning content. Four occurrences
/// in a short thinking tail identify the control-token attractor without
/// applying the broad phrase-loop heuristic to ordinary prose or code.
pub const CHATML_THINK_CYCLE_WINDOW: usize = 32;
pub const CHATML_THINK_CYCLE_MIN_OPENERS: usize = 4;

/// Content-phase loop detection. Catches the post-`</think>` agentic
/// degeneration mode where the model emits the same sentence over
/// and over (observed 2026-04-26 against Claude Code: "I see I've
/// been creating Cargo.toml files but the user hasn't given me a
/// task. Let me wait for their instructions." × 12). LZ penalty
/// at strength 0.2 nudges but doesn't cure once the attractor is
/// established — we need a hard stop.
///
/// Periods extend up to 64 tokens because content-phase loops are
/// full sentences (20-50 tokens), not 4-20-token fence-narration
/// fragments. MIN_TOKENS is higher (96) to give legitimate prose
/// breathing room — three contiguous identical 30-token sentences
/// in a 280-token window is overwhelmingly degenerate.
///
/// Caveat: legitimate structured-code generation also produces
/// period-N repetition. Examples that false-positive:
/// - Chess board JS init: `{color:BLACK,type:'P'},` × 8 (period ~10)
/// - Arrays of identical empty-row HTML cells, multiplication
///   tables, JSON arrays of similar objects, repeated CSS rule
///   blocks, etc.
///
/// **Gating**: this watchdog is OFF by default. Models with a known
/// prose-attractor failure mode (Qwen3.5-35B-A3B + Claude-Code agentic
/// sessions) opt in via MODEL.toml `[behavior].enable_loop_watchdog =
/// true`. The flag is read at boot and stored in
/// [`set_enable_loop_watchdog`] / [`enable_loop_watchdog`].
pub const CONTENT_LOOP_MIN_TOKENS: u32 = 96;
pub const CONTENT_LOOP_CHECK_STRIDE: u32 = 16;
pub const CONTENT_LOOP_PERIOD_MIN: usize = 8;
pub const CONTENT_LOOP_PERIOD_MAX: usize = 64;
pub const CONTENT_LOOP_MIN_REPEATS: usize = 3;
pub const CONTENT_LOOP_SCAN_WINDOW: usize = 280;
/// Min repeats for the digit-normalized content-loop path. Stricter
/// than `CONTENT_LOOP_MIN_REPEATS` (3) because numeric normalization
/// collapses more sequences to a common period — requiring 4 keeps a
/// legitimate 3-item numbered list (`- item 1\n- item 2\n- item 3`)
/// from tripping the hard stop.
pub const CONTENT_LOOP_NORM_MIN_REPEATS: usize = 4;
/// Always-on last-resort loop guard. Unlike the ordinary content watchdog,
/// this requires many *contiguous* byte-identical cycles and therefore remains
/// active when the more sensitive watchdog is disabled for code/HTML output.
/// It exists to bound pathological generations such as runaway period-10
/// token attractors that would otherwise run for tens of thousands of tokens.
pub const CATASTROPHIC_LOOP_MIN_TOKENS: usize = 512;
pub const CATASTROPHIC_LOOP_PERIOD_MIN: usize = 4;
pub const CATASTROPHIC_LOOP_PERIOD_MAX: usize = 64;
pub const CATASTROPHIC_LOOP_MIN_REPEATS: usize = 32;
/// Sentinel substituted for every numeric token in the normalized
/// scan-window tail. `u32::MAX` can never collide with a real vocab id
/// (Qwen3.6 vocab ≤ ~152k), and the `(t as usize) < mask.len()` bound
/// in the classifier means a stray real `u32::MAX` would degrade to
/// "structural", never a false numeric — safe either way.
pub const NUMERIC_SENTINEL: u32 = u32::MAX;

static ENABLE_LOOP_WATCHDOG: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static ENABLE_CATASTROPHIC_LOOP_WATCHDOG: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
static ENABLE_THINKING_LOOP_WATCHDOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Set for each model load from the resolved behavior and CLI override.
/// Model swaps must refresh this value rather than inheriting the first model's
/// policy for the lifetime of the process.
pub fn set_enable_loop_watchdog(enabled: bool) {
    ENABLE_LOOP_WATCHDOG.store(enabled, std::sync::atomic::Ordering::SeqCst);
}

/// Read the per-model loop-watchdog flag set at load. Defaults to
/// `false` until `set_enable_loop_watchdog` runs (load order: weights →
/// behavior plumbing → scheduler start).
pub fn enable_loop_watchdog() -> bool {
    ENABLE_LOOP_WATCHDOG.load(std::sync::atomic::Ordering::Relaxed)
}

/// Stop cause reported for a guard that ends generation. The catastrophic
/// content-loop guard is named separately because it stays on when the
/// configurable watchdog is off.
pub fn stop_cause_for(intervention: &str) -> &'static str {
    if intervention == "catastrophic_content_loop_watchdog" {
        "catastrophic_guard"
    } else {
        "watchdog"
    }
}

/// Record a generation-ending watchdog intervention and remember why the
/// sequence stopped, so it finishes as `"stop"` rather than `"length"`.
/// The first cause wins.
pub fn record_watchdog_stop(a: &mut super::ActiveSeq, intervention: &'static str) {
    crate::openai::record_shared_intervention(
        a.intervention_ledger.as_ref(),
        intervention,
        "terminate_generation",
    );
    a.stop_cause.get_or_insert(stop_cause_for(intervention));
}

/// Enable or disable the last-resort content-loop guard.
///
/// This guard is enabled independently of MODEL.toml, but an operator's
/// explicit `--disable-loop-watchdog` override disables it together with the
/// more sensitive configured detector. Unlike the model-scoped setting, the
/// operator override may be refreshed when a model is swapped in-process.
pub fn set_enable_catastrophic_loop_watchdog(enabled: bool) {
    ENABLE_CATASTROPHIC_LOOP_WATCHDOG.store(enabled, std::sync::atomic::Ordering::SeqCst);
}

pub fn enable_catastrophic_loop_watchdog() -> bool {
    ENABLE_CATASTROPHIC_LOOP_WATCHDOG.load(std::sync::atomic::Ordering::Relaxed)
}

/// Set once at startup from the explicit CLI override. The thinking-loop
/// watchdog historically ran unconditionally, so its fail-open default is
/// `true` until startup plumbing resolves the operator setting.
pub fn set_enable_thinking_loop_watchdog(enabled: bool) {
    let _ = ENABLE_THINKING_LOOP_WATCHDOG.set(enabled);
}

pub fn enable_thinking_loop_watchdog() -> bool {
    // deepseek_v41: the Python engine has no thinking-loop watchdog.
    *ENABLE_THINKING_LOOP_WATCHDOG.get().unwrap_or(&true) && !crate::dsv41::serving()
}

// ── Grammar forced-token fast-path (xgrammar Tier 3b) ───────────────────────

/// Resolved kill-switch for the grammar forced-token (Coalescence)
/// fast-path. Computed once on first read from the environment.
///
/// The fast-path emits a token directly — skipping the model sample and
/// the vocab-wide bitmask fill — only when the active tool-call grammar
/// admits exactly one legal next token (xgrammar's `forced_token`
/// guarantees a single-bit mask). Output is therefore bit-identical to
/// the sampled path, so the fast-path is **on by default**.
///
/// `ATLAS_DISABLE_FORCED_TOKEN=1` (or `true`) forces it off — a
/// kill-switch should a future grammar/matcher regression ever make the
/// forced-token guarantee unsafe. This mirrors the env-var bisection
/// gates already used in `phase_continue_prefills.rs` /
/// `mod_helpers.rs`; a MODEL.toml `[behavior]` flag was not used because
/// the `ModelBehavior` struct lives in the `atlas-kernels` crate, which
/// this change deliberately does not touch.
static FORCED_TOKEN_FASTPATH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Pure parse of the `ATLAS_DISABLE_FORCED_TOKEN` env value into the
/// resolved "fast-path enabled" boolean. Split out of
/// [`forced_token_fastpath_enabled`] so the parsing rule is unit-testable
/// without touching the process-wide `OnceLock`.
///
/// `None` (env unset) → enabled. A truthy value (`"1"` / `"true"`,
/// case-insensitive, surrounding whitespace ignored) → disabled.
/// Everything else (empty, `"0"`, `"false"`, junk) → enabled.
fn parse_forced_token_fastpath(env: Option<&str>) -> bool {
    match env {
        Some(v) => {
            let v = v.trim();
            !(v == "1" || v.eq_ignore_ascii_case("true"))
        }
        None => true,
    }
}

/// Whether the grammar forced-token fast-path is enabled (default
/// `true`; disabled by `ATLAS_DISABLE_FORCED_TOKEN=1`/`true`).
pub fn forced_token_fastpath_enabled() -> bool {
    *FORCED_TOKEN_FASTPATH.get_or_init(|| {
        parse_forced_token_fastpath(std::env::var("ATLAS_DISABLE_FORCED_TOKEN").ok().as_deref())
    })
}

/// Per-model tunables for the always-on decode-time watchdogs. Sourced
/// from MODEL.toml `[behavior]`; the field defaults reproduce the
/// historical hardcoded constants exactly, so a model that sets nothing
/// behaves byte-identically to before parameterization.
#[derive(Debug, Clone, Copy)]
pub struct WatchdogParams {
    /// Thinking-loop watchdog: substring-occurrence count that trips a
    /// forced `</think>`. Default 3 (`THINK_LOOP_MIN_REPEATS`).
    pub think_loop_min_repeats: usize,
    /// Thinking-loop watchdog: trailing-token scan window. Default 160
    /// (`THINK_LOOP_SCAN_WINDOW`).
    pub think_loop_scan_window: usize,
    /// F2 confidence-run early-stop enabled. Default `true`. Set false in
    /// MODEL.toml for models whose deterministic code drafting trips the
    /// heuristic.
    pub confidence_early_stop: bool,
    /// F2 confidence run length before arming forced `</think>`.
    /// Default 30 (`CONFIDENCE_RUN_LIMIT`).
    pub confidence_run_length: u32,
    /// Fuzzy-repetition detector Hamming tolerance divisor: a
    /// `pattern_len`-token window tolerates `pattern_len / div`
    /// mismatches. Default 12 (~8%).
    pub fuzzy_repeat_tolerance_div: usize,
    /// Cap on free-text tokens between successive `<tool_call>` opens in
    /// `tool_choice=auto`. Default [`MAX_INTER_TOOL_PROSE`].
    pub max_inter_tool_prose: u32,
    /// When a degeneration watchdog fires, attempt a terminal rollback to the
    /// last well-formed boundary to trim the degenerate tail before stopping
    /// (`finished = true`); does not re-steer into continued generation.
    /// Default `true`. See [`super::rollback::rollback_to_boundary`].
    pub rollback_resteer: bool,
}

/// Historical-default watchdog tunables — the single source of truth.
/// Each field equals the constant the watchdog used before
/// parameterization, so an unset MODEL.toml `[behavior]` is byte-exact.
/// `CONFIDENCE_RUN_LIMIT` now lives in the sibling `confidence` module
/// (F2 helper extraction); referenced here as the historical default.
const DEFAULT_WATCHDOG_PARAMS: WatchdogParams = WatchdogParams {
    think_loop_min_repeats: THINK_LOOP_MIN_REPEATS,
    think_loop_scan_window: THINK_LOOP_SCAN_WINDOW,
    confidence_early_stop: true,
    confidence_run_length: super::confidence::CONFIDENCE_RUN_LIMIT,
    fuzzy_repeat_tolerance_div: 12,
    max_inter_tool_prose: MAX_INTER_TOOL_PROSE,
    rollback_resteer: true,
};

impl Default for WatchdogParams {
    fn default() -> Self {
        DEFAULT_WATCHDOG_PARAMS
    }
}

static WATCHDOG_PARAMS: std::sync::OnceLock<WatchdogParams> = std::sync::OnceLock::new();

/// Set once at startup from the resolved `ModelBehavior`. Idempotent.
pub fn set_watchdog_params(p: WatchdogParams) {
    let _ = WATCHDOG_PARAMS.set(p);
}

/// Refreshable per-model boolean. Unlike the older watchdog `OnceLock`, this
/// must follow every model load: retaining the first model's value would leak
/// Qwen's EOS semantics into another target (or fail to enable it when Qwen is
/// loaded second).
struct RefreshableBehaviorFlag(std::sync::atomic::AtomicBool);

impl RefreshableBehaviorFlag {
    const fn new(value: bool) -> Self {
        Self(std::sync::atomic::AtomicBool::new(value))
    }

    fn store(&self, value: bool) {
        self.0.store(value, std::sync::atomic::Ordering::Release);
    }

    fn load(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }
}

static HONOR_EOS_INSIDE_THINKING: RefreshableBehaviorFlag = RefreshableBehaviorFlag::new(false);

pub fn set_honor_eos_inside_thinking(enabled: bool) {
    HONOR_EOS_INSIDE_THINKING.store(enabled);
}

pub fn honor_eos_inside_thinking() -> bool {
    HONOR_EOS_INSIDE_THINKING.load()
}

/// Minimum thinking tokens before `</think>` may be emitted (0 = off).
/// `ATLAS_MIN_THINKING_TOKENS`. Counters the observed temp-0 anomaly where
/// the model closes thinking on token 1 for some prompts and blurts a wrong
/// answer (instruction expert.01 = 10 total output tokens). Serving-layer
/// only: masks the </think> logit, weights untouched.
pub fn min_thinking_floor() -> u32 {
    static FLOOR: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *FLOOR.get_or_init(|| {
        std::env::var("ATLAS_MIN_THINKING_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Read the per-model watchdog tunables. Returns the historical-default
/// `WatchdogParams` until `set_watchdog_params` runs — so unit tests and
/// any pre-boot caller see exactly the old hardcoded constants.
pub fn watchdog_params() -> WatchdogParams {
    *WATCHDOG_PARAMS.get().unwrap_or(&DEFAULT_WATCHDOG_PARAMS)
}

/// `mask[id] == true` iff token `id` decodes to a pure ASCII-digit run
/// (optionally one leading space). Built once at startup from the
/// tokenizer; drives the digit-normalized content-loop path. Fail-open:
/// never set (or build failed) → normalized path inert, the exact
/// detector is unaffected.
static NUMERIC_TOKEN_MASK: std::sync::OnceLock<std::sync::Arc<[bool]>> = std::sync::OnceLock::new();

/// Set once at startup from the resolved tokenizer. Idempotent.
pub fn set_numeric_token_mask(mask: std::sync::Arc<[bool]>) {
    let _ = NUMERIC_TOKEN_MASK.set(mask);
}

/// Read the numeric-token mask. `None` until `set_numeric_token_mask`
/// runs — callers must treat `None` as "normalized path disabled".
pub fn numeric_token_mask() -> Option<std::sync::Arc<[bool]>> {
    NUMERIC_TOKEN_MASK.get().cloned()
}

/// `mask[id] == true` iff token `id` decodes to text ending in a
/// well-formed generation boundary — a newline, or sentence-ending
/// punctuation (`.`, `!`, `?`) optionally followed by a closing quote
/// or whitespace. Built once at startup from the tokenizer; drives
/// [`super::rollback::rollback_to_boundary`]'s boundary search.
/// Fail-open: never set → rollback finds no boundary and the watchdog
/// falls back to its hard stop.
static BOUNDARY_TOKEN_MASK: std::sync::OnceLock<std::sync::Arc<[bool]>> =
    std::sync::OnceLock::new();

/// Set once at startup from the resolved tokenizer. Idempotent.
pub fn set_boundary_token_mask(mask: std::sync::Arc<[bool]>) {
    let _ = BOUNDARY_TOKEN_MASK.set(mask);
}

/// Read the boundary-token mask. `None` until `set_boundary_token_mask`
/// runs — callers must treat `None` as "no boundary info available".
pub fn boundary_token_mask() -> Option<std::sync::Arc<[bool]>> {
    BOUNDARY_TOKEN_MASK.get().cloned()
}

/// F2 (2026-04-26): cap on free-text tokens between successive
/// `<tool_call>` opens when `tool_choice="auto"`. The grammar FSM
/// in `auto` mode (`grammar/compile_tools.rs::tag_policy`) sets
/// `at_least_one=false` and, unless the request sent
/// `parallel_tool_calls=false`, `stop_after_first=false`, so the matcher
/// never becomes terminal after the first tool call — the model can emit
/// prose↔tool↔prose↔tool indefinitely. The shared default is deliberately
/// plan-sized: coding agents can legitimately inspect results and explain a
/// multi-step edit before the next call. Counted across non-thinking,
/// non-tool-body tokens only.
pub const MAX_INTER_TOOL_PROSE: u32 = atlas_kernels::DEFAULT_MAX_INTER_TOOL_PROSE;

/// Resolve CLI > environment > MODEL.toml/default. Zero means disabled;
/// keeping it as zero would fire the `prose_tokens > max` check on token 1.
pub fn resolve_max_inter_tool_prose(toml: u32, env: Option<u32>, cli: Option<u32>) -> u32 {
    match cli.or(env).unwrap_or(toml) {
        0 => u32::MAX,
        value => value,
    }
}

#[cfg(test)]
mod inter_tool_prose_tests {
    use super::{MAX_INTER_TOOL_PROSE, resolve_max_inter_tool_prose};

    #[test]
    fn default_is_plan_sized_and_matches_kernels() {
        assert_eq!(
            MAX_INTER_TOOL_PROSE,
            atlas_kernels::DEFAULT_MAX_INTER_TOOL_PROSE
        );
        const { assert!(MAX_INTER_TOOL_PROSE >= 2048) };
    }

    #[test]
    fn precedence_and_zero_disable_are_explicit() {
        assert_eq!(resolve_max_inter_tool_prose(384, None, None), 384);
        assert_eq!(resolve_max_inter_tool_prose(384, Some(8192), None), 8192);
        assert_eq!(
            resolve_max_inter_tool_prose(384, Some(8192), Some(4096)),
            4096
        );
        assert_eq!(resolve_max_inter_tool_prose(384, None, Some(0)), u32::MAX);
    }
}

/// Return `true` iff some contiguous subsequence of length
/// `p ∈ [THINK_LOOP_PERIOD_MIN, THINK_LOOP_PERIOD_MAX]` appears
/// `THINK_LOOP_MIN_REPEATS`+ times in the last
/// `THINK_LOOP_SCAN_WINDOW` tokens.
///
/// Designed to catch the Qwen3.5-35B fence-narration attractor where
/// the loop has a stable phrase body (` \`\`\`bash cd X && cargo test
/// \`\`\` `) but varying connective prefixes (`Running:` /
/// `Executing:` / `I need to run:`). A strict "contiguous
/// periodic repeat" detector misses these; a substring-occurrence
/// counter catches them.
pub fn detect_thinking_token_loop(tokens: &[u32]) -> bool {
    let wp = watchdog_params();
    detect_token_loop(
        tokens,
        THINK_LOOP_MIN_TOKENS as usize,
        THINK_LOOP_PERIOD_MIN,
        THINK_LOOP_PERIOD_MAX,
        wp.think_loop_min_repeats,
        wp.think_loop_scan_window,
    )
}

/// Detect a repeated `<|im_start|>` attractor inside a thinking span.
///
/// `current` supports the serial sampler, which runs thinking policy before
/// pushing the sampled token. Speculative emission passes `None` because its
/// current token is already present in `tokens`.
pub fn detect_chatml_thinking_cycle(tokens: &[u32], current: Option<u32>, im_start: u32) -> bool {
    let retained = CHATML_THINK_CYCLE_WINDOW.saturating_sub(usize::from(current.is_some()));
    let prior = tokens
        .iter()
        .rev()
        .take(retained)
        .filter(|&&tok| tok == im_start)
        .count();
    prior + usize::from(current == Some(im_start)) >= CHATML_THINK_CYCLE_MIN_OPENERS
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InitialThinkingState {
    pub inside_thinking: bool,
    pub think_ended: bool,
    pub think_just_ended: bool,
    pub thinking_budget: Option<u32>,
    pub thinking_tokens: u32,
    pub force_end_thinking: bool,
    pub in_code_fence: bool,
    pub remaining: usize,
    /// Serial thinking emission consumes a zero-budget requested `</think>`
    /// and an in-thinking EOS without adding either to output or streaming it.
    pub first_consumed_silently: bool,
    /// The first sampled token closed a requested reasoning span.  Model-aware
    /// callers use this to take the same unfed-token SSM boundary snapshot as
    /// ordinary decode after constructing the `ActiveSeq`.
    pub closed_on_first: bool,
}

impl InitialThinkingState {
    /// Whether the first sampled token belongs in output, stream token events,
    /// and response logprobs. All three surfaces must use the same decision.
    pub fn first_is_visible(self, spontaneous_think: bool) -> bool {
        !spontaneous_think && !self.first_consumed_silently
    }

    /// A silently consumed first token closed reasoning and therefore needs a
    /// structural stream event before any following content token.
    pub fn publishes_hidden_boundary(self) -> bool {
        self.closed_on_first && self.first_consumed_silently
    }
}

/// Derive every thinking field affected by the token sampled directly from
/// prefill logits, before an `ActiveSeq` exists.  Requested-thinking templates
/// enter the reasoning span in the prompt, while a spontaneous `<think>` token
/// opens the span but is itself control.  A first `</think>` closes a requested
/// span immediately and remains the unfed `last_token`, matching serial decode.
#[allow(clippy::too_many_arguments)]
pub fn initial_thinking_state(
    enable_thinking: bool,
    spontaneous_think: bool,
    first: u32,
    think_end_token: Option<u32>,
    eos_tokens: &[u32],
    request_thinking_budget: Option<u32>,
    spontaneous_think_budget: u32,
    code_fence_token: Option<u32>,
    max_tokens: usize,
    honor_eos_inside_thinking: bool,
    eos_suppressed_outside_thinking: bool,
) -> InitialThinkingState {
    let requested_inside = enable_thinking && think_end_token.is_some();
    let explicit_close_on_first = requested_inside && think_end_token == Some(first);
    let suppressed_eos_on_first =
        requested_inside && !explicit_close_on_first && eos_tokens.contains(&first);
    let implicit_close_on_first =
        suppressed_eos_on_first && honor_eos_inside_thinking && !eos_suppressed_outside_thinking;
    let closed_on_first = explicit_close_on_first || implicit_close_on_first;
    let thinking_budget = if spontaneous_think {
        Some(spontaneous_think_budget)
    } else {
        request_thinking_budget
    };
    // Serial thinking counts a suppressed EOS as reasoning content and feeds
    // it back as `last_token`; only the close delimiter itself is control.
    let thinking_tokens = u32::from(requested_inside && !explicit_close_on_first);
    let inside_thinking = (requested_inside || spontaneous_think) && !closed_on_first;
    let think_ended =
        closed_on_first || (!spontaneous_think && !enable_thinking && think_end_token.is_some());
    let force_end_thinking =
        inside_thinking && thinking_budget.is_some_and(|budget| thinking_tokens >= budget);
    let in_code_fence = requested_inside && !closed_on_first && code_fence_token == Some(first);
    let first_consumed_silently =
        suppressed_eos_on_first || (closed_on_first && thinking_budget == Some(0));
    // Reasoning content, a close delimiter, and a suppressed spontaneous
    // opener are free from the content budget. `max_output_tokens` still caps
    // the absolute response envelope independently.
    let remaining = if requested_inside || spontaneous_think {
        max_tokens
    } else {
        max_tokens.saturating_sub(1)
    };

    InitialThinkingState {
        inside_thinking,
        think_ended,
        think_just_ended: closed_on_first,
        thinking_budget,
        thinking_tokens,
        force_end_thinking,
        in_code_fence,
        remaining,
        first_consumed_silently,
        closed_on_first,
    }
}

/// Content-phase analogue of [`detect_thinking_token_loop`] — fires
/// when the model emits the same sentence over and over after
/// `</think>` has closed (the Claude-Code 2026-04-26 degeneration).
pub fn detect_content_token_loop(tokens: &[u32]) -> bool {
    detect_token_loop(
        tokens,
        CONTENT_LOOP_MIN_TOKENS as usize,
        CONTENT_LOOP_PERIOD_MIN,
        CONTENT_LOOP_PERIOD_MAX,
        CONTENT_LOOP_MIN_REPEATS,
        CONTENT_LOOP_SCAN_WINDOW,
    )
}

/// Detect only a catastrophic, perfectly periodic tail.
///
/// The ordinary watchdog counts repeated substrings anywhere in a short scan
/// window and is intentionally configurable because generated tables, CSS,
/// and source code can resemble that shape.  This safety net is deliberately
/// much stricter: the final 32 complete periods must be contiguous and exactly
/// equal.  That makes it suitable as an always-on bound even when the ordinary
/// watchdog is disabled.
pub fn detect_catastrophic_content_loop(tokens: &[u32]) -> bool {
    let n = tokens.len();
    if n < CATASTROPHIC_LOOP_MIN_TOKENS {
        return false;
    }

    let max_period = CATASTROPHIC_LOOP_PERIOD_MAX.min(n / CATASTROPHIC_LOOP_MIN_REPEATS);
    for period in CATASTROPHIC_LOOP_PERIOD_MIN..=max_period {
        let repeated_len = period * CATASTROPHIC_LOOP_MIN_REPEATS;
        let tail = &tokens[n - repeated_len..];
        let needle = &tail[..period];
        if tail
            .chunks_exact(period)
            .all(|candidate| candidate == needle)
        {
            return true;
        }
    }
    false
}

/// Digit-normalized content-loop detector. Maps every numeric token in
/// the scan-window TAIL to [`NUMERIC_SENTINEL`], then period-matches —
/// catching the Qwen3.6-27B greedy degeneration where the line template
/// is fixed (`- B(46) = N\n`) but the integer payload varies each line,
/// so the exact [`detect_content_token_loop`] never fires.
///
/// Allocates only the ≤ `CONTENT_LOOP_SCAN_WINDOW` tail copy; the full
/// history is never normalized. FP mitigation: stricter
/// `CONTENT_LOOP_NORM_MIN_REPEATS`, and the matched period must contain
/// BOTH a sentinel (numeric) and a non-sentinel (structural) token —
/// pure-number columns and pure-prose loops are left to the exact path.
pub fn detect_content_token_loop_normalized(tokens: &[u32], mask: &[bool]) -> bool {
    let n = tokens.len();
    if n < CONTENT_LOOP_MIN_TOKENS as usize {
        return false;
    }
    let tail_start = n.saturating_sub(CONTENT_LOOP_SCAN_WINDOW);
    let is_numeric = |t: u32| (t as usize) < mask.len() && mask[t as usize];
    // Map numeric tokens to the sentinel AND run-length-collapse
    // consecutive sentinels to ONE. Qwen3.6 is digit-level
    // (`104509868777` → 12 single-digit tokens, `273508641` → 9), so a
    // bare 1:1 map would leave variable-length sentinel runs and the
    // period would still vary line to line. Collapsing makes
    // `- B(<digits>) = <digits>\n` identical regardless of digit count.
    let mut norm: Vec<u32> = Vec::with_capacity(CONTENT_LOOP_SCAN_WINDOW);
    for &t in &tokens[tail_start..] {
        if is_numeric(t) {
            if norm.last() != Some(&NUMERIC_SENTINEL) {
                norm.push(NUMERIC_SENTINEL);
            }
        } else {
            norm.push(t);
        }
    }
    // No qualifying period can exist without both kinds of token —
    // cheap early-out before the O(period·window) scan.
    let has_sentinel = norm.contains(&NUMERIC_SENTINEL);
    let has_struct = norm.iter().any(|&t| t != NUMERIC_SENTINEL);
    if !has_sentinel || !has_struct {
        return false;
    }
    detect_token_loop_with_period(
        &norm,
        CONTENT_LOOP_PERIOD_MIN,
        CONTENT_LOOP_PERIOD_MAX,
        CONTENT_LOOP_NORM_MIN_REPEATS,
        CONTENT_LOOP_SCAN_WINDOW,
    )
}

/// Substring-occurrence loop detector used by both the thinking
/// and content phases. Returns `true` iff some contiguous
/// subsequence of length `p ∈ [period_min, period_max]` appears
/// `min_repeats`+ times in the last `scan_window` tokens of `tokens`.
pub fn detect_token_loop(
    tokens: &[u32],
    min_tokens: usize,
    period_min: usize,
    period_max: usize,
    min_repeats: usize,
    scan_window: usize,
) -> bool {
    let n = tokens.len();
    if n < min_tokens {
        return false;
    }
    let tail_start = n.saturating_sub(scan_window);
    let tail = &tokens[tail_start..];
    for period in period_min..=period_max {
        if tail.len() < period * min_repeats {
            continue;
        }
        let needle = &tail[tail.len() - period..];
        let mut count = 0usize;
        let mut pos = 0usize;
        while pos + period <= tail.len() {
            if &tail[pos..pos + period] == needle {
                count += 1;
                if count >= min_repeats {
                    return true;
                }
                pos += period; // non-overlapping
            } else {
                pos += 1;
            }
        }
    }
    false
}

/// Like [`detect_token_loop`] but only accepts a periodic match whose
/// `needle` (the period-length window) contains BOTH a
/// [`NUMERIC_SENTINEL`] and a non-sentinel token. Used exclusively by
/// the digit-normalized path so a pure-number column or a pure-prose
/// repeat does not trip here (those remain the exact detector's job).
/// The `< min_tokens` guard is the caller's `CONTENT_LOOP_MIN_TOKENS`
/// check, so this takes only the tail/period rules.
///
/// ~20 lines duplicate `detect_token_loop`'s scan: the per-period
/// needle predicate needs the matched window, which `detect_token_loop`
/// (`-> bool`) hides. Duplicating is lower-risk than adding a closure
/// param to a function with 13 existing tests + the exact call site;
/// the exact detector stays byte-identical (regression-tested).
fn detect_token_loop_with_period(
    tokens: &[u32],
    period_min: usize,
    period_max: usize,
    min_repeats: usize,
    scan_window: usize,
) -> bool {
    let n = tokens.len();
    let tail_start = n.saturating_sub(scan_window);
    let tail = &tokens[tail_start..];
    for period in period_min..=period_max {
        if tail.len() < period * min_repeats {
            continue;
        }
        let needle = &tail[tail.len() - period..];
        let needle_ok =
            needle.contains(&NUMERIC_SENTINEL) && needle.iter().any(|&t| t != NUMERIC_SENTINEL);
        if !needle_ok {
            continue;
        }
        let mut count = 0usize;
        let mut pos = 0usize;
        while pos + period <= tail.len() {
            if &tail[pos..pos + period] == needle {
                count += 1;
                if count >= min_repeats {
                    return true;
                }
                pos += period; // non-overlapping
            } else {
                pos += 1;
            }
        }
    }
    false
}

// F2 confidence-run + code-fence pure helpers (`toggle_code_fence`,
// `confidence_run_step`, `should_inject_think_end` + their constants)
// were moved to `confidence.rs` to keep this file ≤500 LoC. They are
// re-exported through the scheduler module so existing `super::*`
// call sites are unaffected.

#[cfg(test)]
#[path = "helpers_tests.rs"]
mod thinking_loop_tests;
