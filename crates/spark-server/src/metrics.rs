// SPDX-License-Identifier: AGPL-3.0-only

//! Prometheus metrics for Atlas Spark.

use std::sync::atomic::{AtomicUsize, Ordering};

use lazy_static::lazy_static;
use prometheus::{
    Gauge, GaugeVec, Histogram, IntCounter, IntCounterVec, IntGauge, register_gauge,
    register_gauge_vec, register_histogram, register_int_counter, register_int_counter_vec,
    register_int_gauge,
};

/// SLO-budgeted prefill interleave metric names (`scheduler::prefill_slo`).
pub const PREFILL_CHUNK_MS_METRIC: &str = "atlas_prefill_chunk_ms";
pub const PREFILL_CHUNK_ERROR_METRIC: &str = "atlas_prefill_chunk_prediction_error_ms";
pub const PREFILL_CHUNK_ABS_ERROR_METRIC: &str = "atlas_prefill_chunk_prediction_abs_error_ms";
pub const DECODE_STEP_MS_METRIC: &str = "atlas_decode_step_ms";
pub const INTERLEAVE_DECISIONS_METRIC: &str = "atlas_prefill_interleave_decisions_total";

lazy_static! {
    pub static ref REQUESTS_TOTAL: IntCounter =
        register_int_counter!("atlas_requests_total", "Total requests processed").unwrap();
    pub static ref REQUESTS_ACTIVE: IntGauge =
        register_int_gauge!("atlas_requests_active", "Currently active requests").unwrap();
    pub static ref TTFT_SECONDS: Histogram = register_histogram!(
        "atlas_time_to_first_token_seconds",
        "Time to first token",
        vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    )
    .unwrap();
    /// Tokens counted AS THEY ARE PRODUCED, in the streaming handler.
    ///
    /// Distinct from `GENERATION_TOKENS_TOTAL`, which only moves when a
    /// request finishes: sampling that alone reads as 0 tok/s for the whole
    /// generation and then one spike, which is not a rate. The dashboard takes
    /// the larger of the two deltas, so blocking requests — which never stream
    /// and legitimately arrive as a lump — still register.
    pub static ref DECODED_TOKENS_TOTAL: IntCounter =
        register_int_counter!("atlas_decoded_tokens_total", "Tokens counted as decoded, per token").unwrap();
    pub static ref GENERATION_TOKENS_TOTAL: IntCounter =
        register_int_counter!("atlas_generation_tokens_total", "Total tokens generated").unwrap();
    // ── HTTP byte accounting (TUI Server Stats; mounted when TUI is active) ──
    //
    // Request side counts declared `Content-Length` header bytes (0 for
    // chunked requests without Content-Length); response side counts bytes
    // actually written through the wrapped body (streaming/SSE included).
    // Both counters stay 0 in `--no-tui` mode because `byte_count` is only
    // mounted when the TUI is enabled.
    pub static ref HTTP_BYTES_IN: IntCounter =
        register_int_counter!("atlas_http_bytes_in_total", "Total HTTP request body bytes")
            .unwrap();
    pub static ref HTTP_BYTES_OUT: IntCounter =
        register_int_counter!("atlas_http_bytes_out_total", "Total HTTP response body bytes")
            .unwrap();
    pub static ref PROMPT_TOKENS_TOTAL: IntCounter =
        register_int_counter!("atlas_prompt_tokens_total", "Total prompt tokens processed")
            .unwrap();
    /// Scheduler-accepted prefill chunks, published after the model call
    /// returns success rather than when the whole request returns.  This is a
    /// host-progress signal: a long prompt can legitimately have no output
    /// token for minutes while still advancing one bounded chunk at a time.
    pub static ref PREFILL_PROGRESS_CHUNKS_TOTAL: IntCounter = register_int_counter!(
        "atlas_prefill_progress_chunks_total",
        "Prompt-prefill chunks accepted by the model and scheduler"
    )
    .unwrap();
    /// Prompt tokens in scheduler-accepted prefill chunks.  Unlike
    /// `atlas_prompt_tokens_total`, this moves before TTFT.
    pub static ref PREFILL_PROGRESS_TOKENS_TOTAL: IntCounter = register_int_counter!(
        "atlas_prefill_progress_tokens_total",
        "Prompt tokens accepted in successful prefill chunks"
    )
    .unwrap();

    // ── Loop-detector telemetry (P5.2, 2026-04-25) ──
    //
    // Track the verdict distribution emitted by `loop_detector::detect`
    // so we can tune thresholds against production traffic instead of
    // single dump fixtures. Labels:
    //   - verdict ∈ {none, hint, suppress}
    //   - channel ∈ {text, tools, combined, n/a (None verdict)}
    //   - spinning ∈ {0, 1} — was Layer-2 spinning detection also active
    pub static ref LOOP_DETECTOR_VERDICTS: IntCounterVec =
        register_int_counter_vec!(
            "atlas_loop_detector_verdicts_total",
            "Loop detector verdicts emitted, by verdict + channel + spinning flag",
            &["verdict", "channel", "spinning"]
        ).unwrap();

    // ── Server-side intervention telemetry (P5.2) ──
    //
    // Track how often the goal-pin reminder + observation-masking
    // fire, so we can correlate intervention frequency with outcome
    // metrics (TTFT, completion length, finish_reason).
    pub static ref THINK_BUDGET_FORCED_CLOSE: IntCounterVec =
        register_int_counter_vec!(
            "flux_think_budget_forced_close_total",
            "Thinking-budget forced closes, by whether the answer nudge was injected",
            &["nudge"]
        ).unwrap();
    pub static ref TASK_PIN_INJECTIONS: IntCounter =
        register_int_counter!(
            "atlas_task_pin_injections_total",
            "Times the verbatim-goal reminder was injected into a request"
        ).unwrap();
    pub static ref OBSERVATION_MASK_ELIDED_BODIES: IntCounter =
        register_int_counter!(
            "atlas_observation_mask_elided_bodies_total",
            "Stale tool-failure bodies replaced with one-line summaries"
        ).unwrap();

    // ── Anthropic translation-drift counter (P5.1) ──
    //
    // Increments whenever the Anthropic→OpenAI translator produces a
    // round-trip diff against the original Anthropic shape. Diffs
    // indicate translation bugs that compound across long agentic
    // sessions. Logging the actual diff is gated behind the
    // ATLAS_DEBUG_TRANSLATION_DRIFT env var (`anthropic/translate.rs`).
    pub static ref ANTHROPIC_TRANSLATION_DRIFTS: IntCounter =
        register_int_counter!(
            "atlas_anthropic_translation_drifts_total",
            "Anthropic ↔ OpenAI translator round-trip mismatches detected"
        ).unwrap();

    // ── Speculative-decode telemetry (A.2 EASD scaffolding) ──
    //
    // Per-K acceptance counters. Enables measuring baseline accept
    // rates across MTP K-paths so we can decide whether EASD
    // activation (per-step D2H of verify logits + entropy gating,
    // arXiv:2512.23765) is worth its cost. EASD itself is gated
    // behind future activation once these baselines are measured.
    pub static ref SPEC_DECODE_VERIFY: IntCounterVec =
        register_int_counter_vec!(
            "atlas_spec_decode_verify_total",
            "MTP draft verify outcomes by K and result",
            &["k", "outcome"]
        ).unwrap();

    // ── SLO-budgeted prefill interleave (`scheduler::prefill_slo`) ──
    //
    // Learned chunk cost per (width bucket, prompt-offset band): the EWMA of
    // measured chunk wall time, the last signed prediction error (measured −
    // predicted by the decision that ran it) and the absolute-error
    // distribution; the decode-step EWMA the gap target leaves room for; and
    // every decision taken while a row decodes.
    pub static ref PREFILL_CHUNK_MS: GaugeVec =
        register_gauge_vec!(
            PREFILL_CHUNK_MS_METRIC,
            "EWMA of measured prefill chunk wall time (ms) per width bucket and offset band",
            &["width", "band"]
        ).unwrap();
    pub static ref PREFILL_CHUNK_PREDICTION_ERROR_MS: GaugeVec =
        register_gauge_vec!(
            PREFILL_CHUNK_ERROR_METRIC,
            "Last prefill chunk prediction error (measured - predicted, ms)",
            &["width", "band"]
        ).unwrap();
    pub static ref PREFILL_CHUNK_PREDICTION_ABS_ERROR_MS: Histogram = register_histogram!(
        PREFILL_CHUNK_ABS_ERROR_METRIC,
        "Absolute prefill chunk prediction error (ms)",
        vec![5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    ).unwrap();
    pub static ref DECODE_STEP_MS: Gauge =
        register_gauge!(
            DECODE_STEP_MS_METRIC,
            "EWMA of decode-only scheduler tick wall time (ms)"
        ).unwrap();
    pub static ref PREFILL_INTERLEAVE_DECISIONS: IntCounterVec =
        register_int_counter_vec!(
            INTERLEAVE_DECISIONS_METRIC,
            "Interleave decisions on ticks with decode rows: target_met, short_lane, starvation, single_chunk, parity, skip, passthrough",
            &["decision"]
        ).unwrap();
    // ── Tool-call telemetry ──
    //
    // Total successful tool calls emitted by the API layer (sum across
    // streaming + blocking). Paired with the "Tool call: name(args)"
    // info log so operators can both grep logs and graph rates.
    // Unlabeled (no `name` label) — high-cardinality tool names would
    // blow up Prometheus cardinality.
    pub static ref TOOL_CALLS_TOTAL: IntCounter =
        register_int_counter!(
            "atlas_tool_calls_total",
            "Total successful tool calls emitted by the server"
        ).unwrap();

    // ── Qwen4 native-batch routing (`spark_model::qwen4_batch_policy`) ──
    //
    // Native C2-C32 decode is the default; every other C2+ frame names its
    // serial reason. Labels are the stable `SerialReason::label` tokens
    // (`none` on native frames), so cardinality is bounded by the enum.
    pub static ref QWEN4_DECODE_FRAMES: IntCounterVec =
        register_int_counter_vec!(
            "atlas_qwen4_decode_frames_total",
            "Qwen4 C2+ decode frame attempts by pre-dispatch route and reason; not successful execution",
            &["route", "reason"]
        ).unwrap();
    pub static ref QWEN4_SERIAL_ROW_STEPS: IntCounterVec =
        register_int_counter_vec!(
            "atlas_qwen4_serial_row_steps_total",
            "Qwen4 serial row-step attempts selected inside C2+ frames, by row reason; not completed row-steps",
            &["reason"]
        ).unwrap();
    pub static ref QWEN4_DECODE_FRAME_OUTCOMES: IntCounterVec =
        register_int_counter_vec!(
            "atlas_qwen4_decode_frame_outcomes_total",
            "Qwen4 routed C2+ decode_batch returned outcomes; not GPU completion or sampling success",
            &["route", "reason", "outcome"]
        ).unwrap();
    pub static ref QWEN4_REQUEST_ROUTES: IntCounterVec =
        register_int_counter_vec!(
            "atlas_qwen4_requests_total",
            "Qwen4 requests admitted with --max-batch-size > 1, by intrinsic route and reason",
            &["route", "reason"]
        ).unwrap();
}

/// Publish one measured prefill chunk of the SLO cost model.
pub fn record_prefill_chunk(width: usize, band: &str, ewma_ms: f64, error_ms: f64) {
    let width = width.to_string();
    PREFILL_CHUNK_MS
        .with_label_values(&[&width, band])
        .set(ewma_ms);
    PREFILL_CHUNK_PREDICTION_ERROR_MS
        .with_label_values(&[&width, band])
        .set(error_ms);
    PREFILL_CHUNK_PREDICTION_ABS_ERROR_MS.observe(error_ms.abs());
}

/// Publish the decode-step EWMA.
pub fn record_decode_step_ms(ms: f64) {
    DECODE_STEP_MS.set(ms);
}

/// Count one interleave decision.
pub fn record_interleave_decision(label: &str) {
    PREFILL_INTERLEAVE_DECISIONS
        .with_label_values(&[label])
        .inc();
}

/// Publish one monotone, request-scoped prefill progress receipt.
///
/// Call only after the model call returns success and the scheduler advances
/// its logical offset.  This does not add a GPU synchronization: it proves
/// host-side forward progress, not that all queued CUDA work has completed. A
/// supervisor may use advancement as evidence that a long request is alive,
/// but must retain an independent hard deadline and another signal (output,
/// health, or a later stream synchronization). In particular, a timeout must
/// never replay the request: attention KV and recurrent state may already have
/// been mutated by every accepted chunk.
pub fn record_prefill_progress(
    session_hash: u64,
    completed_tokens: usize,
    prompt_tokens: usize,
    chunk_tokens: usize,
) {
    // A zero-length or out-of-range receipt could keep an external stall
    // watchdog alive without proving any forward motion.  Reject it in release
    // builds too; this telemetry is part of the liveness contract, not merely
    // debug instrumentation.
    if chunk_tokens == 0 || completed_tokens > prompt_tokens {
        tracing::error!(
            "PREFILL_PROGRESS_INVALID schema=1 session={session_hash:016x} completed_tokens={completed_tokens} prompt_tokens={prompt_tokens} chunk_tokens={chunk_tokens}"
        );
        return;
    }
    PREFILL_PROGRESS_CHUNKS_TOTAL.inc();
    let chunk_tokens_u64 = u64::try_from(chunk_tokens).unwrap_or(u64::MAX);
    PREFILL_PROGRESS_TOKENS_TOTAL.inc_by(chunk_tokens_u64);
    compat_metrics::add_prompt_tokens(chunk_tokens_u64);
    tracing::info!(
        "PREFILL_PROGRESS schema=1 session={session_hash:016x} completed_tokens={completed_tokens} prompt_tokens={prompt_tokens} chunk_tokens={chunk_tokens}"
    );
}

/// Compact `vllm:*` OpenMetrics compatibility counters and gauges.
///
/// Rendered ahead of `atlas_*` metrics on `GET /metrics` so benchmark harnesses
/// inspecting the leading response bytes discover vLLM-compatible gauges,
/// cache config metadata, and speculative-decoding counters immediately.
pub mod compat_metrics {
    use std::fmt::Write as _;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    const KV_BLOCKS_FREE_UNSET: usize = usize::MAX;

    static CACHE_BLOCK_SIZE: AtomicUsize = AtomicUsize::new(0);
    static CACHE_NUM_GPU_BLOCKS: AtomicUsize = AtomicUsize::new(0);
    static CACHE_KV_SIZE_TOKENS: AtomicUsize = AtomicUsize::new(0);
    static NUM_REQUESTS_RUNNING: AtomicUsize = AtomicUsize::new(0);
    static NUM_REQUESTS_WAITING: AtomicUsize = AtomicUsize::new(0);
    static KV_BLOCKS_FREE: AtomicUsize = AtomicUsize::new(KV_BLOCKS_FREE_UNSET);
    static GENERATION_TOKENS: AtomicU64 = AtomicU64::new(0);
    static PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
    static REQUEST_SUCCESS: AtomicU64 = AtomicU64::new(0);
    static NUM_PREEMPTIONS: AtomicU64 = AtomicU64::new(0);
    static SPEC_NUM_DRAFTS: AtomicU64 = AtomicU64::new(0);
    static SPEC_NUM_DRAFT_TOKENS: AtomicU64 = AtomicU64::new(0);
    static SPEC_NUM_ACCEPTED_TOKENS: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone, Copy, Debug)]
    pub(super) struct Snapshot {
        pub running: usize,
        pub waiting: usize,
        pub block_size: usize,
        pub total_blocks: usize,
        pub kv_tokens: usize,
        pub free_blocks: usize,
        pub generation_tokens: u64,
        pub prompt_tokens: u64,
        pub request_success: u64,
        pub preemptions: u64,
        pub spec_drafts: u64,
        pub spec_draft_tokens: u64,
        pub spec_accepted_tokens: u64,
        pub prefix_queries: u64,
        pub prefix_hits: u64,
    }

    pub(super) fn kv_usage_fraction(total_blocks: usize, free_blocks: usize) -> f64 {
        if total_blocks == 0 || free_blocks == KV_BLOCKS_FREE_UNSET {
            return 0.0;
        }
        let used = total_blocks.saturating_sub(free_blocks.min(total_blocks));
        (used as f64 / total_blocks as f64).clamp(0.0, 1.0)
    }

    pub(super) fn render_snapshot(s: &Snapshot) -> String {
        let kv_usage = kv_usage_fraction(s.total_blocks, s.free_blocks);
        let mut out = String::with_capacity(1_280);
        let _ = write!(
            out,
            "# TYPE vllm:num_requests_running gauge\n\
             vllm:num_requests_running {}\n\
             # TYPE vllm:num_requests_waiting gauge\n\
             vllm:num_requests_waiting {}\n\
             # TYPE vllm:kv_cache_usage_perc gauge\n\
             vllm:kv_cache_usage_perc {kv_usage:.6}\n",
            s.running, s.waiting,
        );
        if s.total_blocks > 0 {
            let _ = write!(
                out,
                "# TYPE vllm:cache_config_info gauge\n\
                 vllm:cache_config_info{{block_size=\"{}\",num_gpu_blocks=\"{}\",kv_cache_size_tokens=\"{}\"}} 1\n",
                s.block_size, s.total_blocks, s.kv_tokens,
            );
        }
        let _ = write!(
            out,
            "# TYPE vllm:generation_tokens_total counter\n\
             vllm:generation_tokens_total {}\n\
             # TYPE vllm:prompt_tokens_total counter\n\
             vllm:prompt_tokens_total {}\n\
             # TYPE vllm:request_success_total counter\n\
             vllm:request_success_total {}\n\
             # TYPE vllm:num_preemptions_total counter\n\
             vllm:num_preemptions_total {}\n\
             # TYPE vllm:spec_decode_num_drafts_total counter\n\
             vllm:spec_decode_num_drafts_total {}\n\
             # TYPE vllm:spec_decode_num_draft_tokens_total counter\n\
             vllm:spec_decode_num_draft_tokens_total {}\n\
             # TYPE vllm:spec_decode_num_accepted_tokens_total counter\n\
             vllm:spec_decode_num_accepted_tokens_total {}\n\
             # TYPE vllm:prefix_cache_queries_total counter\n\
             vllm:prefix_cache_queries_total {}\n\
             # TYPE vllm:prefix_cache_hits_total counter\n\
             vllm:prefix_cache_hits_total {}\n",
            s.generation_tokens,
            s.prompt_tokens,
            s.request_success,
            s.preemptions,
            s.spec_drafts,
            s.spec_draft_tokens,
            s.spec_accepted_tokens,
            s.prefix_queries,
            s.prefix_hits,
        );
        out
    }

    /// Record paged KV cache geometry reported in `vllm:cache_config_info`.
    pub fn set_cache_config(block_size: usize, num_gpu_blocks: usize, kv_cache_size_tokens: usize) {
        CACHE_BLOCK_SIZE.store(block_size, Ordering::Relaxed);
        CACHE_NUM_GPU_BLOCKS.store(num_gpu_blocks, Ordering::Relaxed);
        CACHE_KV_SIZE_TOKENS.store(kv_cache_size_tokens, Ordering::Relaxed);
    }

    /// Update running (`active + prefilling`) and waiting (`pending`) request gauges.
    pub fn set_scheduler_counts(running: usize, waiting: usize) {
        NUM_REQUESTS_RUNNING.store(running, Ordering::Relaxed);
        NUM_REQUESTS_WAITING.store(waiting, Ordering::Relaxed);
    }

    /// Update only the waiting request gauge from the receiver thread without
    /// clobbering the scheduler thread's running count.
    pub(crate) fn set_waiting_count(waiting: usize) {
        NUM_REQUESTS_WAITING.store(waiting, Ordering::Relaxed);
    }

    /// Update the current number of free paged KV cache blocks.
    pub fn set_kv_blocks_free(free_blocks: usize) {
        KV_BLOCKS_FREE.store(free_blocks, Ordering::Relaxed);
    }

    /// Add committed generation tokens to `vllm:generation_tokens_total`.
    pub fn add_generation_tokens(count: u64) {
        if count > 0 {
            GENERATION_TOKENS.fetch_add(count, Ordering::Relaxed);
        }
    }

    /// Add processed prompt tokens to `vllm:prompt_tokens_total`.
    pub fn add_prompt_tokens(count: u64) {
        if count > 0 {
            PROMPT_TOKENS.fetch_add(count, Ordering::Relaxed);
        }
    }

    /// Increment `vllm:request_success_total` when a sequence completes.
    pub fn inc_request_success() {
        REQUEST_SUCCESS.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `vllm:num_preemptions_total` when a sequence is evicted under KV pressure.
    pub fn inc_preemptions() {
        NUM_PREEMPTIONS.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one speculative verification step's drafted and accepted token counts.
    pub fn record_spec_decode(draft_tokens: u64, accepted_tokens: u64) {
        if draft_tokens > 0 {
            SPEC_NUM_DRAFTS.fetch_add(1, Ordering::Relaxed);
            SPEC_NUM_DRAFT_TOKENS.fetch_add(draft_tokens, Ordering::Relaxed);
            SPEC_NUM_ACCEPTED_TOKENS
                .fetch_add(accepted_tokens.min(draft_tokens), Ordering::Relaxed);
        }
    }

    /// Render the compact `vllm:*` Prometheus text block for prepending to `/metrics`.
    #[must_use]
    pub fn render_vllm_openmetrics(prefix_queries: u64, prefix_hits: u64) -> String {
        let snapshot = Snapshot {
            running: NUM_REQUESTS_RUNNING.load(Ordering::Relaxed),
            waiting: NUM_REQUESTS_WAITING.load(Ordering::Relaxed),
            block_size: CACHE_BLOCK_SIZE.load(Ordering::Relaxed),
            total_blocks: CACHE_NUM_GPU_BLOCKS.load(Ordering::Relaxed),
            kv_tokens: CACHE_KV_SIZE_TOKENS.load(Ordering::Relaxed),
            free_blocks: KV_BLOCKS_FREE.load(Ordering::Relaxed),
            generation_tokens: GENERATION_TOKENS.load(Ordering::Relaxed),
            prompt_tokens: PROMPT_TOKENS.load(Ordering::Relaxed),
            request_success: REQUEST_SUCCESS.load(Ordering::Relaxed),
            preemptions: NUM_PREEMPTIONS.load(Ordering::Relaxed),
            spec_drafts: SPEC_NUM_DRAFTS.load(Ordering::Relaxed),
            spec_draft_tokens: SPEC_NUM_DRAFT_TOKENS.load(Ordering::Relaxed),
            spec_accepted_tokens: SPEC_NUM_ACCEPTED_TOKENS.load(Ordering::Relaxed),
            prefix_queries,
            prefix_hits,
        };
        render_snapshot(&snapshot)
    }
}

static IN_FLIGHT_INGRESS: AtomicUsize = AtomicUsize::new(0);

/// Number of HTTP requests currently in ingress (tokenizing / building sampling
/// state) before being enqueued onto the scheduler's pending queue.
pub fn in_flight_ingress() -> usize {
    IN_FLIGHT_INGRESS.load(Ordering::Acquire)
}

/// Holds one count on [`IN_FLIGHT_INGRESS`] while a request is being parsed and
/// tokenized prior to `scheduler.submit(...)`.
pub struct IngressGuard {
    active: bool,
}

impl Default for IngressGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl IngressGuard {
    pub fn new() -> Self {
        IN_FLIGHT_INGRESS.fetch_add(1, Ordering::AcqRel);
        Self { active: true }
    }

    pub fn finish(&mut self) {
        if self.active {
            self.active = false;
            IN_FLIGHT_INGRESS.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl Drop for IngressGuard {
    fn drop(&mut self) {
        if self.active {
            IN_FLIGHT_INGRESS.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Holds one count on [`REQUESTS_ACTIVE`] for as long as it is alive.
///
/// THE BUG THIS FIXES: `chat::completions` incremented the gauge and then had
/// TEN early returns — input validation, image admission, tool-prompt
/// prepending — none of which decremented. Every rejected request leaked a
/// count permanently, so the gauge only ever grew on bad input, and
/// `shutdown::drain_in_flight` waits on exactly that gauge: a server that had
/// seen malformed requests could never finish draining.
///
/// One `inc()` against six `dec()`s spread over four modules is not a thing
/// anyone can keep right by reading it, which is why this is a guard.
pub struct ActiveRequestGuard {
    released: bool,
}

impl Default for ActiveRequestGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl ActiveRequestGuard {
    pub fn new() -> Self {
        REQUESTS_ACTIVE.inc();
        IN_FLIGHT_INGRESS.fetch_add(1, Ordering::AcqRel);
        Self { released: false }
    }

    /// Hand the count to a path that decrements it ITSELF.
    ///
    /// The streaming and blocking dispatchers outlive this scope and already
    /// own their own `dec()` — a guard that also fired would double-decrement,
    /// and a gauge driven negative makes the drain wait forever on a count
    /// that can never reach zero. Consuming `self` means the transfer is a
    /// statement at the call site, not a comment someone has to notice.
    pub fn release(mut self) {
        if !self.released {
            self.released = true;
            IN_FLIGHT_INGRESS.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        if !self.released {
            IN_FLIGHT_INGRESS.fetch_sub(1, Ordering::AcqRel);
            REQUESTS_ACTIVE.dec();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slo_interleave_metrics_are_registered_under_their_names() {
        record_prefill_chunk(512, "0-8k", 340.0, -10.0);
        record_decode_step_ms(48.0);
        record_interleave_decision("skip");
        let names: Vec<String> = prometheus::gather()
            .iter()
            .map(|family| family.get_name().to_string())
            .collect();
        for name in [
            "atlas_prefill_chunk_ms",
            "atlas_prefill_chunk_prediction_error_ms",
            "atlas_prefill_chunk_prediction_abs_error_ms",
            "atlas_decode_step_ms",
            "atlas_prefill_interleave_decisions_total",
        ] {
            assert!(names.iter().any(|n| n == name), "missing {name}");
        }
        assert_eq!(
            PREFILL_CHUNK_MS.with_label_values(&["512", "0-8k"]).get(),
            340.0
        );
        assert_eq!(
            PREFILL_CHUNK_PREDICTION_ERROR_MS
                .with_label_values(&["512", "0-8k"])
                .get(),
            -10.0
        );
        assert!(
            PREFILL_INTERLEAVE_DECISIONS
                .with_label_values(&["skip"])
                .get()
                >= 1
        );
    }

    #[test]
    fn prefill_progress_counts_only_real_forward_motion() {
        let chunks_before = PREFILL_PROGRESS_CHUNKS_TOTAL.get();
        let tokens_before = PREFILL_PROGRESS_TOKENS_TOTAL.get();

        record_prefill_progress(0x1234, 17, 100, 17);
        assert_eq!(PREFILL_PROGRESS_CHUNKS_TOTAL.get(), chunks_before + 1);
        assert_eq!(PREFILL_PROGRESS_TOKENS_TOTAL.get(), tokens_before + 17);

        record_prefill_progress(0x1234, 17, 100, 0);
        record_prefill_progress(0x1234, 101, 100, 1);
        assert_eq!(PREFILL_PROGRESS_CHUNKS_TOTAL.get(), chunks_before + 1);
        assert_eq!(PREFILL_PROGRESS_TOKENS_TOTAL.get(), tokens_before + 17);
    }

    #[test]
    fn vllm_compat_openmetrics_stays_compact_and_leading() {
        let rendered = compat_metrics::render_snapshot(&compat_metrics::Snapshot {
            running: 8,
            waiting: 12,
            block_size: 16,
            total_blocks: 16_400,
            kv_tokens: 262_400,
            free_blocks: 4_100,
            generation_tokens: 1_234_567,
            prompt_tokens: 9_876_543,
            request_success: 4_200,
            preemptions: 3,
            spec_drafts: 120_000,
            spec_draft_tokens: 480_000,
            spec_accepted_tokens: 310_000,
            prefix_queries: 5_000,
            prefix_hits: 3_500,
        });
        assert!(
            rendered.len() < 1_600,
            "vllm compat block must stay well under 2000-byte probe: {} bytes",
            rendered.len()
        );
        assert!(
            rendered.starts_with("# TYPE vllm:"),
            "vllm compat block must start at byte 0: {rendered:?}"
        );
        assert!(rendered[..64].contains("vllm:"));
        for required in [
            "vllm:num_requests_running 8\n",
            "vllm:num_requests_waiting 12\n",
            "vllm:kv_cache_usage_perc 0.750000\n",
            "vllm:cache_config_info{block_size=\"16\",num_gpu_blocks=\"16400\",kv_cache_size_tokens=\"262400\"} 1\n",
            "vllm:generation_tokens_total 1234567\n",
            "vllm:prompt_tokens_total 9876543\n",
            "vllm:request_success_total 4200\n",
            "vllm:num_preemptions_total 3\n",
            "vllm:spec_decode_num_drafts_total 120000\n",
            "vllm:spec_decode_num_draft_tokens_total 480000\n",
            "vllm:spec_decode_num_accepted_tokens_total 310000\n",
            "vllm:prefix_cache_queries_total 5000\n",
            "vllm:prefix_cache_hits_total 3500\n",
        ] {
            assert!(
                rendered.contains(required),
                "missing {required:?} in:\n{rendered}"
            );
        }
    }

    #[test]
    fn vllm_compat_kv_cache_usage_perc_bounds_and_zero_total() {
        assert_eq!(compat_metrics::kv_usage_fraction(0, 0), 0.0);
        assert_eq!(compat_metrics::kv_usage_fraction(0, 50), 0.0);
        assert_eq!(compat_metrics::kv_usage_fraction(100, usize::MAX), 0.0);
        assert_eq!(compat_metrics::kv_usage_fraction(100, 100), 0.0);
        assert_eq!(compat_metrics::kv_usage_fraction(100, 150), 0.0);
        assert_eq!(compat_metrics::kv_usage_fraction(100, 0), 1.0);
        let half = compat_metrics::kv_usage_fraction(100, 50);
        assert!((0.0..=1.0).contains(&half));
        assert!((half - 0.5).abs() < 1e-12);

        // cache_config_info is omitted when total_blocks == 0.
        let unconfigured = compat_metrics::render_snapshot(&compat_metrics::Snapshot {
            running: 0,
            waiting: 0,
            block_size: 0,
            total_blocks: 0,
            kv_tokens: 0,
            free_blocks: 0,
            generation_tokens: 0,
            prompt_tokens: 0,
            request_success: 0,
            preemptions: 0,
            spec_drafts: 0,
            spec_draft_tokens: 0,
            spec_accepted_tokens: 0,
            prefix_queries: 0,
            prefix_hits: 0,
        });
        assert!(!unconfigured.contains("vllm:cache_config_info"));
        assert!(unconfigured.contains("vllm:kv_cache_usage_perc 0.000000\n"));
    }

    #[test]
    fn vllm_compat_public_api_updates_render_output() {
        compat_metrics::set_cache_config(16, 64, 1_024);
        compat_metrics::set_kv_blocks_free(32);
        compat_metrics::set_scheduler_counts(2, 1);
        compat_metrics::add_generation_tokens(5);
        compat_metrics::add_prompt_tokens(10);
        compat_metrics::inc_request_success();
        compat_metrics::inc_preemptions();
        compat_metrics::record_spec_decode(4, 3);

        let rendered = compat_metrics::render_vllm_openmetrics(9, 4);
        assert!(rendered.len() < 1_600);
        assert!(rendered.starts_with("# TYPE vllm:"));
        assert!(rendered.contains("vllm:prefix_cache_queries_total 9\n"));
        assert!(rendered.contains("vllm:prefix_cache_hits_total 4\n"));
        assert!(rendered.contains("vllm:cache_config_info{block_size=\"16\",num_gpu_blocks=\"64\",kv_cache_size_tokens=\"1024\"} 1\n"));
    }
}
