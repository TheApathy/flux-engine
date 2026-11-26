// SPDX-License-Identifier: AGPL-3.0-only

//! Device top-k sampling route (`ATLAS_SAMPLER_DEVICE_TOPK=1|auto`, SPEED_PLAN W1).
//!
//! A row takes the device route only when every host intervention that
//! `process_seq_logits` would apply to it can lower logits or mask all but one
//! id, never raise a logit or read the whole row. The device then returns the
//! `m = top_k + E` best candidates (E bounds the ids those interventions may
//! lower), the host builds a -inf row holding them, and the SAME intervention
//! and sampler code runs on it. The proof that this is output-identical lives
//! in `spark_runtime::sampler::device_topk`; its unit tests check it on random
//! rows with ties, penalties and edits.
//!
//! Everything else is a receipted host route: `route=host_full reason=...`.
//! Under `ATLAS_SAMPLER_DEVICE_TOPK=1` (`Strict`), a request whose `top_k` the
//! device cannot serve is refused at admission. Under
//! `ATLAS_SAMPLER_DEVICE_TOPK=auto` (`Auto`), all requests are admitted and
//! ineligible requests transparently fall back to the host sampler path. A
//! device kernel failure fails the request; it never silently falls back.

use super::*;
use spark_runtime::sampler::device_topk;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Set by [`device_topk_startup_receipt`] when `ATLAS_SAMPLER_DEVICE_TOPK=auto`
/// runs against a model that does not implement `topk_candidates_bf16`.
static AUTO_MODEL_UNSUPPORTED: AtomicBool = AtomicBool::new(false);

fn auto_model_unsupported() -> bool {
    AUTO_MODEL_UNSUPPORTED.load(Ordering::Relaxed)
}

/// Upper bound on ids the plain path's own interventions may lower on one row
/// beyond the reflection list: min-thinking floor (1), `</think>` + `<think>`
/// masks after thinking (2), `<tool_call>` mask or -12 bias (1).
const FIXED_LOWERED_IDS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum TopkRoute {
    /// `f2`: the row is inside the F2 window, so its confidence decision is
    /// certified on device (or, when uncertain, taken exactly on the host
    /// from the full row) before sampling.
    Device {
        m: usize,
        temperature: f32,
        f2: bool,
    },
    Host(&'static str),
}

/// Decide the route for `a`'s next plain-decode row. Pure: reads state only.
pub(super) fn device_topk_route(
    a: &ActiveSeq,
    logits_fp32: bool,
    adaptive_sampling: bool,
    reflection_suppress_ids: &[u32],
) -> TopkRoute {
    if let Some(reason) = host_only_reason(a, logits_fp32, adaptive_sampling) {
        return TopkRoute::Host(reason);
    }
    let params = a.sampling_params(a.temperature);
    let extra = reflection_suppress_ids.len() + FIXED_LOWERED_IDS;
    let f2 = a.inside_thinking && super::think_spec_accept::f2_window_active(a);
    if f2 && reflection_suppress_ids.len() + 1 > spark_runtime::sampler::f2_certify::F2_EDITS_MAX {
        return TopkRoute::Host("f2_edit_budget");
    }
    match device_topk::candidate_budget(&params, &a.output_tokens, extra) {
        Ok(m) => TopkRoute::Device {
            m,
            temperature: params.temperature,
            f2,
        },
        Err(reason) => TopkRoute::Host(reason),
    }
}

fn host_only_reason(
    a: &ActiveSeq,
    logits_fp32: bool,
    adaptive_sampling: bool,
) -> Option<&'static str> {
    let checks: [(bool, &'static str); 9] = [
        (auto_model_unsupported(), "model_unsupported"),
        (logits_fp32, "fp32_logits"),
        (adaptive_sampling, "adaptive_sampling"),
        (
            crate::dsv41::serving() || crate::dsv41::repetition::active(),
            "dsv41",
        ),
        (a.top_logprobs.is_some(), "logprobs"),
        (
            super::decode_logits_seq::logits_fnv_requires_row(),
            "logits_fnv",
        ),
        (bootstrap_trace_enabled(), "bootstrap_trace"),
        (a.grammar_state.is_some() && !a.inside_thinking, "grammar"),
        (
            a.inside_thinking && crate::scheduler::think_efficiency_config().is_active(),
            "think_efficiency",
        ),
    ];
    checks
        .iter()
        .find(|(hit, _)| *hit)
        .map(|(_, reason)| *reason)
}

/// Whether `a` should take the device top-k route under the configured
/// `ATLAS_SAMPLER_DEVICE_TOPK` mode (`Off`, `Strict`, or `Auto`).
pub(super) fn seq_uses_device_topk(
    model: &dyn Model,
    a: &ActiveSeq,
    logits_fp32: bool,
    adaptive_sampling: bool,
    reflection_suppress_ids: &[u32],
) -> bool {
    seq_uses_device_topk_with_mode(
        model.device_topk_supported(),
        a,
        logits_fp32,
        adaptive_sampling,
        reflection_suppress_ids,
        device_topk::device_topk_mode(),
    )
}

/// Pure helper for [`seq_uses_device_topk`] parameterized by `mode` and model support.
pub(super) fn seq_uses_device_topk_with_mode(
    model_supported: bool,
    a: &ActiveSeq,
    logits_fp32: bool,
    adaptive_sampling: bool,
    reflection_suppress_ids: &[u32],
    mode: device_topk::DeviceTopkMode,
) -> bool {
    mode.is_enabled()
        && model_supported
        && matches!(
            device_topk_route(a, logits_fp32, adaptive_sampling, reflection_suppress_ids),
            TopkRoute::Device { .. }
        )
}

fn bootstrap_trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_SPEC_BOOTSTRAP_TRACE").ok().as_deref() == Some("1"))
}

/// The device half: candidates for one BF16 row, as a vocab-length f32 row.
/// `Ok(None)` when the row has fewer than `m` finite logits (the sampler's
/// empty-row fallback may then need the real non-finite values, so the caller
/// copies the full row). Errors are request-fatal.
pub(super) fn device_topk_sparse_row(
    model: &dyn Model,
    row: DevicePtr,
    m: usize,
    temperature: f32,
) -> Result<Option<Vec<f32>>> {
    let candidates = model.topk_candidates(row, temperature, m)?;
    anyhow::ensure!(
        candidates.len() <= m,
        "device top-k returned {} candidates for m={m}",
        candidates.len()
    );
    // Fewer than m finite logits: after up to E host masks the sampler could
    // see no finite value and fall back to the raw row's non-finite entries,
    // which the sparse row does not carry. Take the full row instead.
    if candidates.len() < m {
        return Ok(None);
    }
    device_topk::sparse_row_from_candidates(model.vocab_size(), &candidates)
        .map(Some)
        .ok_or_else(|| anyhow::anyhow!("device top-k returned a token id outside the vocabulary"))
}

/// Whether plain-decode rows get a route decision and a receipt: the route
/// is on, or `ATLAS_SAMPLER_ROUTE_STATS=1` asks for the per-row timing on the
/// control arm (route off, every row `host_full reason=route_off`).
pub(super) fn route_receipts_enabled() -> bool {
    static EXPLICIT_STATS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    device_topk::device_topk_enabled()
        || *EXPLICIT_STATS
            .get_or_init(|| std::env::var("ATLAS_SAMPLER_ROUTE_STATS").ok().as_deref() == Some("1"))
}

/// Routes for a plain-decode batch, or `None` when receipts are off (the
/// historical path runs untouched).
pub(super) fn plain_batch_routes(
    active: &[ActiveSeq],
    logits_fp32: bool,
    adaptive_sampling: bool,
    reflection_suppress_ids: &[u32],
) -> Option<Vec<TopkRoute>> {
    plain_batch_routes_with_mode(
        active,
        logits_fp32,
        adaptive_sampling,
        reflection_suppress_ids,
        device_topk::device_topk_mode(),
        route_receipts_enabled(),
    )
}

/// Pure helper for [`plain_batch_routes`] parameterized by `mode` and `receipts`.
pub(super) fn plain_batch_routes_with_mode(
    active: &[ActiveSeq],
    logits_fp32: bool,
    adaptive_sampling: bool,
    reflection_suppress_ids: &[u32],
    mode: device_topk::DeviceTopkMode,
    receipts: bool,
) -> Option<Vec<TopkRoute>> {
    if !receipts {
        return None;
    }
    if !mode.is_enabled() {
        return Some(vec![TopkRoute::Host("route_off"); active.len()]);
    }
    let routes: Vec<TopkRoute> = active
        .iter()
        .map(|a| device_topk_route(a, logits_fp32, adaptive_sampling, reflection_suppress_ids))
        .collect();
    // One full-batch D2H serves any host row; a mixed batch takes it for all.
    if routes.iter().all(|r| matches!(r, TopkRoute::Device { .. })) {
        Some(routes)
    } else {
        Some(
            routes
                .into_iter()
                .map(|r| match r {
                    TopkRoute::Device { .. } => TopkRoute::Host("mixed_batch"),
                    host => host,
                })
                .collect(),
        )
    }
}

/// One plain-decode row fetched for sampling.
pub(super) enum FetchedRow {
    /// Device candidates as a vocab-length f32 row (-inf elsewhere), and the
    /// row's F2 decision when it is inside the F2 window.
    Sparse(Vec<f32>, Option<bool>),
    /// The row had fewer than `m` finite logits; the sampler's fallback reads
    /// the real non-finite values, so this is the full BF16 row.
    FullBf16(Vec<u8>),
}

/// Fetch every row before any row is sampled, so a device or copy failure
/// fails the batch before any sequence state has moved.
pub(super) fn fetch_device_rows(
    model: &dyn Model,
    logits: DevicePtr,
    routes: &[TopkRoute],
    active: &[ActiveSeq],
    think_end_token: Option<u32>,
    reflection_suppress_ids: &[u32],
) -> Result<Vec<FetchedRow>> {
    let vocab = model.vocab_size();
    let row_bytes = vocab * 2;
    routes
        .iter()
        .zip(active)
        .enumerate()
        .map(|(i, (route, a))| {
            let TopkRoute::Device { m, temperature, f2 } = *route else {
                anyhow::bail!("fetch_device_rows called with a host row {i}");
            };
            let row = logits.offset(i * row_bytes);
            let Some(sparse) = device_topk_sparse_row(model, row, m, temperature)? else {
                let mut full = vec![0u8; row_bytes];
                model.copy_logits_to_host(row, &mut full)?;
                return Ok(FetchedRow::FullBf16(full));
            };
            let decision = if f2 {
                let edits = super::decode_logits_seq::pre_f2_edits(
                    a,
                    think_end_token,
                    reflection_suppress_ids,
                    vocab,
                );
                Some(f2_decision(model, row, &edits, row_bytes)?)
            } else {
                None
            };
            Ok(FetchedRow::Sparse(sparse, decision))
        })
        .collect()
}

/// The F2 decision for one edited row: certified from device statistics, or,
/// when the bound cannot separate it from the threshold, the plain path's
/// test run verbatim on the full row copied to the host.
pub(super) fn f2_decision(
    model: &dyn Model,
    row: DevicePtr,
    edits: &[(u32, spark_runtime::sampler::f2_certify::PreF2Edit)],
    row_bytes: usize,
) -> Result<bool> {
    use spark_runtime::sampler::f2_certify;
    let stats = model.edited_sum_exp(row, edits)?;
    if let Some(decision) = f2_certify::certify(&stats) {
        F2_CERTIFIED.fetch_add(1, Ordering::Relaxed);
        return Ok(decision);
    }
    F2_HOST_FULL.fetch_add(1, Ordering::Relaxed);
    let mut full = vec![0u8; row_bytes];
    model.copy_logits_to_host(row, &mut full)?;
    Ok(host_f2_from_bf16(&full, edits))
}

/// The plain path's F2 test on a full BF16 row after `edits`.
pub(super) fn host_f2_from_bf16(
    row_bf16: &[u8],
    edits: &[(u32, spark_runtime::sampler::f2_certify::PreF2Edit)],
) -> bool {
    let mut x: Vec<f32> = row_bf16
        .chunks_exact(2)
        .map(|b| bf16_to_f32(b[0], b[1]))
        .collect();
    for &(id, edit) in edits {
        edit.apply(&mut x[id as usize]);
    }
    spark_runtime::sampler::f2_certify::host_f2_confident(&x)
}

// ── Receipts ──

static DEVICE_ROWS: AtomicU64 = AtomicU64::new(0);
static HOST_ROWS: AtomicU64 = AtomicU64::new(0);
static DEVICE_NS: AtomicU64 = AtomicU64::new(0);
static HOST_NS: AtomicU64 = AtomicU64::new(0);
static F2_CERTIFIED: AtomicU64 = AtomicU64::new(0);
static F2_HOST_FULL: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StartupOutcome {
    Disabled,
    Engaged { mode_label: &'static str },
    AutoHostOnly,
}

pub(super) fn startup_receipt_decision(
    model_supported: bool,
    mode: device_topk::DeviceTopkMode,
) -> Result<StartupOutcome> {
    if !mode.is_enabled() {
        return Ok(StartupOutcome::Disabled);
    }
    if !model_supported {
        if mode.is_strict() {
            anyhow::bail!(
                "ATLAS_SAMPLER_DEVICE_TOPK=1 but this model cannot run topk_candidates_bf16; refusing to start"
            );
        }
        return Ok(StartupOutcome::AutoHostOnly);
    }
    let mode_label = if mode.is_strict() { "strict" } else { "auto" };
    Ok(StartupOutcome::Engaged { mode_label })
}

/// One-time load receipt. Called once the model is up; refuses to serve under
/// `ATLAS_SAMPLER_DEVICE_TOPK=1` (`Strict`) when the model cannot run the
/// kernel, and degrades cleanly to host-only under `ATLAS_SAMPLER_DEVICE_TOPK=auto`.
pub fn device_topk_startup_receipt(model: &dyn Model) -> Result<()> {
    match startup_receipt_decision(
        model.device_topk_supported(),
        device_topk::device_topk_mode(),
    )? {
        StartupOutcome::Disabled => {
            AUTO_MODEL_UNSUPPORTED.store(false, Ordering::Relaxed);
        }
        StartupOutcome::AutoHostOnly => {
            AUTO_MODEL_UNSUPPORTED.store(true, Ordering::Relaxed);
            tracing::info!("SAMPLER_DEVICE_TOPK_AUTO_HOST_ONLY mode=auto reason=model_unsupported");
        }
        StartupOutcome::Engaged { mode_label } => {
            AUTO_MODEL_UNSUPPORTED.store(false, Ordering::Relaxed);
            tracing::info!(
                "SAMPLER_DEVICE_TOPK_ENGAGED k_max={} m_max={} tie_order=value_desc_id_asc canonical_ties=1 mode={mode_label}",
                device_topk::DEVICE_TOPK_K_MAX,
                device_topk::DEVICE_TOPK_M_MAX
            );
        }
    }
    Ok(())
}

/// Maximum number of recent sessions tracked for route-change receipt deduplication.
const LAST_ROUTE_MAX_SESSIONS: usize = 4096;

/// O(1) FIFO-bounded session -> last route reason table.
///
/// Updating an existing session never grows the eviction ring; inserting a fresh
/// session once full evicts the oldest entry so memory stays strictly bounded
/// without clearing active sessions' dedup state all at once.
struct BoundedSessionRoutes {
    map: std::collections::HashMap<u64, &'static str>,
    order: std::collections::VecDeque<u64>,
    max_entries: usize,
}

impl BoundedSessionRoutes {
    fn new(max_entries: usize) -> Self {
        Self {
            map: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
            max_entries,
        }
    }

    /// Record `reason` for `session_hash`; returns `true` when the session is
    /// new or its reason changed since the previous call.
    fn record_changed(&mut self, session_hash: u64, reason: &'static str) -> bool {
        if let Some(prev) = self.map.get_mut(&session_hash) {
            if *prev == reason {
                return false;
            }
            *prev = reason;
            return true;
        }
        if self.max_entries == 0 {
            return true;
        }
        while self.map.len() >= self.max_entries {
            let Some(oldest) = self.order.pop_front() else {
                self.map.clear();
                break;
            };
            self.map.remove(&oldest);
        }
        self.map.insert(session_hash, reason);
        self.order.push_back(session_hash);
        true
    }
}

/// Per-row route receipt: logged when a session's route or reason changes
/// (so every request shows its routes without one line per token), plus
/// running per-route sample-time totals.
pub(super) fn note_route(a: &ActiveSeq, route: TopkRoute, elapsed: std::time::Duration) {
    let ns = elapsed.as_nanos().min(u128::from(u64::MAX)) as u64;
    let (label, reason) = match route {
        TopkRoute::Device { f2, .. } => {
            DEVICE_ROWS.fetch_add(1, Ordering::Relaxed);
            DEVICE_NS.fetch_add(ns, Ordering::Relaxed);
            ("device_topk", if f2 { "eligible_f2" } else { "eligible" })
        }
        TopkRoute::Host(reason) => {
            HOST_ROWS.fetch_add(1, Ordering::Relaxed);
            HOST_NS.fetch_add(ns, Ordering::Relaxed);
            ("host_full", reason)
        }
    };
    static LAST: std::sync::OnceLock<parking_lot::Mutex<BoundedSessionRoutes>> =
        std::sync::OnceLock::new();
    let changed = LAST
        .get_or_init(|| parking_lot::Mutex::new(BoundedSessionRoutes::new(LAST_ROUTE_MAX_SESSIONS)))
        .lock()
        .record_changed(a.session_hash, reason);
    if changed {
        let (d, h) = (
            DEVICE_ROWS.load(Ordering::Relaxed),
            HOST_ROWS.load(Ordering::Relaxed),
        );
        let us = |total: &AtomicU64, rows: u64| {
            total.load(Ordering::Relaxed) as f64 / 1000.0 / rows.max(1) as f64
        };
        tracing::info!(
            "SAMPLER_DEVICE_TOPK session={:016x} idx={} route={label} reason={reason} device_rows={d} device_us_per_row={:.1} host_rows={h} host_us_per_row={:.1} f2_certified={} f2_host_full={}",
            a.session_hash,
            a.output_tokens.len(),
            us(&DEVICE_NS, d),
            us(&HOST_NS, h),
            F2_CERTIFIED.load(Ordering::Relaxed),
            F2_HOST_FULL.load(Ordering::Relaxed),
        );
    }
}

#[cfg(test)]
#[path = "device_topk_route_tests.rs"]
mod tests;

#[cfg(test)]
mod bounded_last_tests {
    use super::*;

    #[test]
    fn bounded_session_routes_dedups_and_evicts_fifo_at_cap() {
        let mut routes = BoundedSessionRoutes::new(3);
        assert!(routes.record_changed(10, "eligible"));
        assert!(!routes.record_changed(10, "eligible"));
        assert!(routes.record_changed(10, "mixed_batch"));
        assert!(!routes.record_changed(10, "mixed_batch"));
        assert_eq!(routes.map.len(), 1);
        assert_eq!(routes.order.len(), 1);

        assert!(routes.record_changed(20, "eligible"));
        assert!(routes.record_changed(30, "eligible"));
        assert_eq!(routes.map.len(), 3);
        assert_eq!(routes.order.len(), 3);

        // Inserting a 4th session evicts only the oldest (10), retaining 20 and 30.
        assert!(routes.record_changed(40, "eligible"));
        assert_eq!(routes.map.len(), 3);
        assert_eq!(routes.order.len(), 3);
        assert!(!routes.record_changed(20, "eligible"));
        assert!(!routes.record_changed(30, "eligible"));
        assert!(!routes.record_changed(40, "eligible"));
        assert!(routes.record_changed(10, "mixed_batch"));
        assert_eq!(routes.map.len(), 3);
        assert_eq!(routes.order.len(), 3);

        // High-cardinality churn stays strictly capped at LAST_ROUTE_MAX_SESSIONS.
        let mut full = BoundedSessionRoutes::new(LAST_ROUTE_MAX_SESSIONS);
        for session in 0..(LAST_ROUTE_MAX_SESSIONS as u64 + 500) {
            assert!(full.record_changed(session, "eligible"));
        }
        assert_eq!(full.map.len(), LAST_ROUTE_MAX_SESSIONS);
        assert_eq!(full.order.len(), LAST_ROUTE_MAX_SESSIONS);
    }
}
