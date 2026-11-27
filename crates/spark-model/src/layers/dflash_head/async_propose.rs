// SPDX-License-Identifier: AGPL-3.0-only

//! ATLAS_DFLASH_ASYNC=1 — async propose‖commit-tail overlap (task #20).
//!
//! ## What overlaps (and what cannot)
//!
//! The DFlash drafter conditions on the TARGET's hidden states captured
//! during verify at layers `[1,16,31,46,61]` of the 64-layer stack — the
//! last capture lands ~95% through the verify forward. So `propose(N+1)`
//! is **data-dependent on `verify(N)` completing**: the "propose N+1 while
//! verify N runs" full overlap the naive pipeline picture suggests is
//! architecturally impossible without severing the drafter's ctx-hidden
//! conditioning (measured elsewhere to collapse acceptance).
//!
//! What CAN overlap is everything the CPU does AFTER the drafter kernels
//! are enqueued: today `run_mtp_propose_multi` blocks the scheduler thread
//! in `forward_block`'s final `synchronize` + drafts D2H, so the step-tail
//! CPU work (STEP_TIMING log, tree-payload drain, scheduler loop, HTTP
//! stream flush, next step's phase-1 verify setup) all runs AFTER the
//! drafter GPU work has drained. This module inverts that: the drafter
//! forward is ENQUEUED on a dedicated CUDA stream (ordered after the
//! default stream via a recorded event) and `propose` returns immediately
//! with a placeholder chain; the REAL drafts are collected (stream sync +
//! γ×4-byte D2H) at the top of the NEXT scheduler step, right before the
//! verify needs the token values. The SSM commit tail (secondary stream)
//! and the step-tail CPU work now run concurrently with the drafter.
//!
//! ## Losslessness
//!
//! Byte-identical off (default). On: the drafter runs the SAME kernels on
//! the SAME inputs (the launch is ordered after all default-stream writes
//! it reads — ctx-append D2Ds, verify captures), so the proposed drafts are
//! bit-identical to the sync path; and drafts only ever PROPOSE — the
//! verify oracle commits solely the target's greedy token. A lost /
//! discarded async propose degrades to an empty draft chain → bootstrap
//! decode (slower, never wrong).
//!
//! ## Safety invariants (shared-scratch discipline)
//!
//! The drafter head owns ONE scratch buffer set, so at most ONE async
//! propose may be in flight; the handle lives on the head
//! (`async_inflight`). Every consumer of the scratch resolves the handle
//! first:
//!   * `collect_async_drafts` (scheduler, top of next step) — the normal
//!     path: sync + D2H → real drafts.
//!   * `propose_drafts` (any subsequent launch) — resolves a stale handle
//!     before touching scratch (sequence died / route changed).
//!   * `free_sequence` — resolves before the per-seq ctx buffers the
//!     in-flight kernels read are freed (use-after-free guard).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::draft_budget::{DflashDraftBudget, clear_proposal_outputs, finalize_proposal};
use super::forward_block::checked_topk_difference;
use super::{BlockDiffusionDraftHead, DflashProposerState};

/// Master gate: `ATLAS_DFLASH_ASYNC=1` (default OFF). Cached.
pub fn dflash_async_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_DFLASH_ASYNC").ok().as_deref() == Some("1"))
}

/// ATLAS_DFLASH_FUSED=1: record the propose-ordering event pre-commit so the
/// drafter runs in parallel with SSM commit + KV reshape (~10ms overlap).
/// Requires ATLAS_DFLASH_ASYNC=1. Cached.
pub fn dflash_fused_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_DFLASH_FUSED").ok().as_deref() == Some("1"))
}

/// Opt-in environment selector for Layer-61 (27B) / Layer-45 (Flash-Next)
/// early-start Stage-1 (`fc` + `hidden_norm`) overlap during target verify's
/// trailing layers + LM head + D2H. Default OFF.
pub const DFLASH_EARLY_STAGE1_ENV: &str = "ATLAS_DFLASH_EARLY_STAGE1";

/// Parse `ATLAS_DFLASH_EARLY_STAGE1`: exact `"1"` (trimmed) enables; default OFF.
#[inline]
#[must_use]
pub fn parse_dflash_early_stage1_enabled(raw: Option<&str>) -> bool {
    raw.map(str::trim) == Some("1")
}

/// Cached gate for `ATLAS_DFLASH_EARLY_STAGE1=1` (default OFF).
#[must_use]
pub fn dflash_early_stage1_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        parse_dflash_early_stage1_enabled(std::env::var(DFLASH_EARLY_STAGE1_ENV).ok().as_deref())
    })
}

/// Returns `true` iff `layer_idx` is the deepest (final) DFlash capture layer
/// in `capture_layers` (e.g. Layer `61` for 27B `[5, 19, 33, 47, 61]` or Layer
/// `45` for Flash-Next `[5, 15, 25, 35, 45]`). Empty `capture_layers` returns `false`.
#[inline]
#[must_use]
pub fn is_final_dflash_capture_layer(layer_idx: usize, capture_layers: &[usize]) -> bool {
    capture_layers
        .iter()
        .copied()
        .max()
        .is_some_and(|final_layer| final_layer == layer_idx)
}

/// Double-buffered device output scratch slot for consecutive async batched
/// (`C >= 2`) DFlash2 proposals, preventing output buffer aliasing across
/// consecutive steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AsyncScratchSlot {
    Primary = 0,
    Secondary = 1,
}

impl AsyncScratchSlot {
    /// Alternate to the opposite slot (`Primary <-> Secondary`).
    #[inline]
    #[must_use]
    pub const fn flip(self) -> Self {
        match self {
            Self::Primary => Self::Secondary,
            Self::Secondary => Self::Primary,
        }
    }

    /// Stable numeric slot index (`0` or `1`) for CUDA graph cache keys.
    #[inline]
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Decode slot index modulo 2.
    #[inline]
    #[must_use]
    pub const fn from_u8(raw: u8) -> Self {
        if raw & 1 == 0 {
            Self::Primary
        } else {
            Self::Secondary
        }
    }
}

/// Pure eligibility predicate for cross-sequence batched (`C >= 2`) async DFlash2
/// proposals on the dedicated propose stream.
#[inline]
#[must_use]
pub fn async_batched_launch_eligible(env: &AsyncEnvEligibility, has_markov: bool) -> bool {
    async_launch_eligible(env, has_markov, false)
}

/// Env-derived disqualifiers, computed once. Every feature here either
/// post-processes drafts on the host inside/after `forward_block` (markov,
/// denoise>1, margin gate, adaptive gamma, cfg-jf splice, debug dumps,
/// kernel profile), builds a tree payload from host drafts (branch /
/// caterpillar / free-slots / ddtree / portfolio), or pre-empts the neural
/// drafter with host-side drafts that early-return from `propose_drafts`
/// before the async bookkeeping (retrieval/SAM, PLD, recycle) — all need the
/// drafts on the host at propose() return, which defeats the deferred
/// collect. The async path only engages on the plain flat-chain neural
/// propose (the production default config).
#[derive(Debug, Clone, Copy)]
pub struct AsyncEnvEligibility {
    pub denoise_steps: usize,
    pub margin_gate_on: bool,
    pub adaptive_gamma: bool,
    pub tree_method: bool,
    pub portfolio: bool,
    pub cfg_jf: bool,
    pub kprofile: bool,
    pub debug_dump: bool,
    /// Retrieval drafting (`ATLAS_DFLASH_RETRIEVAL=1` / `ATLAS_DFLASH_SAM=1`):
    /// pre-empts the neural drafter with host-side drafts and early-returns
    /// from `propose_drafts` BEFORE the async launch/collect bookkeeping, so a
    /// prior in-flight async propose is never resolved — violating the
    /// single-in-flight scratch invariant (measured: silent serve death at
    /// drafter init, variants.md 2026-07-18).
    ///
    /// `ATLAS_DFLASH_SAM_ASYNC=1` opts out of this disqualifier. It is safe
    /// when the mtp_step collect (which always runs before propose) resolves the
    /// prior in-flight handle — the retrieval early-return in propose_drafts
    /// only fires AFTER collect has already synced the handle. The disqualifier
    /// is kept as the default because 2026-07-18 crash on GB10+SM120 was
    /// empirically real; the opt-in enables testing SAM+ASYNC coexistence.
    pub retrieval: bool,
    /// `ATLAS_DFLASH_PLD=1`: same host-draft early-return class as retrieval.
    pub pld: bool,
    /// `ATLAS_DFLASH_RECYCLE=1`: same host-draft early-return class.
    pub recycle: bool,
    /// `ATLAS_DFLASH_DEEPLOOP=1`: multi-pass denoise with GPU-side commit-all.
    /// Removes the inter-pass host D2H, making `denoise_steps > 1` async-eligible.
    pub deeploop: bool,
}

impl AsyncEnvEligibility {
    pub fn from_env() -> Self {
        Self::from_lookup(|k| std::env::var(k).ok(), super::kernel_profile_enabled())
    }

    fn from_lookup(mut get: impl FnMut(&str) -> Option<String>, kprofile: bool) -> Self {
        let is_one = |value: Option<String>| value.as_deref() == Some("1");
        let denoise_steps = get("ATLAS_DFLASH_DENOISE_STEPS")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let margin_gate_on = get("ATLAS_DFLASH_MARGIN_GATE")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0)
            > 0.0;
        let adaptive_gamma = is_one(get("ATLAS_DFLASH_ADAPTIVE_GAMMA"));
        let ddtree_method_active = get("ATLAS_DFLASH_METHOD").as_deref() == Some("ddtree");
        // ATLAS_DFLASH_BRANCH, CATERPILLAR, and ATLAS_DFLASH_METHOD=ddtree
        // post-process drafts on the host after forward_block without async
        // collection support. Only ATLAS_DFLASH_FREE_SLOTS uses the deferred
        // top-K path (enqueue_topk_on_stream + collect) and stays async-eligible.
        let tree_method = is_one(get("ATLAS_DFLASH_BRANCH"))
            || is_one(get("ATLAS_DFLASH_CATERPILLAR"))
            || ddtree_method_active;
        let portfolio = is_one(get("ATLAS_DFLASH_PORTFOLIO"));
        let cfg_jf = is_one(get("ATLAS_DFLASH_CFG_JF"));
        let debug_dump = is_one(get("ATLAS_DFLASH_DEBUG_DUMP"))
            || is_one(get("ATLAS_DFLASH_DEBUG_DUMP_FULL"))
            || is_one(get("ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS"));
        // ATLAS_DFLASH_SAM_ASYNC=1 overrides this: when the mtp_step
        // collect always precedes propose, the retrieval early-return in
        // propose_drafts cannot leave a stale in-flight handle (collect
        // already synced it). See field-level comment for safety argument.
        let retrieval = (is_one(get("ATLAS_DFLASH_RETRIEVAL")) || is_one(get("ATLAS_DFLASH_SAM")))
            && !is_one(get("ATLAS_DFLASH_SAM_ASYNC"));
        let pld = is_one(get("ATLAS_DFLASH_PLD"));
        let recycle = is_one(get("ATLAS_DFLASH_RECYCLE"));
        let deeploop = is_one(get("ATLAS_DFLASH_DEEPLOOP"));
        Self {
            denoise_steps,
            margin_gate_on,
            adaptive_gamma,
            tree_method,
            portfolio,
            cfg_jf,
            // Shared with forward_block: ATLAS_FULL_PROFILE=1 also turns the
            // propose profiler on, and profiling requires the synchronous path
            // (async defers the syncs the per-kernel timers depend on), so the
            // async-eligibility check below must see the same value.
            kprofile,
            debug_dump,
            retrieval,
            pld,
            recycle,
            deeploop,
        }
    }

    /// Cached process-wide instance (all inputs are process-lifetime envs).
    pub fn cached() -> &'static Self {
        static CACHED: std::sync::OnceLock<AsyncEnvEligibility> = std::sync::OnceLock::new();
        CACHED.get_or_init(Self::from_env)
    }
}

/// Pure launch-eligibility decision (unit-tested, GPU-free).
///
/// `has_markov`: the checkpoint ships a Markov head (host-side sequential
/// re-sampling of the block after D2H — incompatible with deferred collect).
/// `grammar_masked`: caller passed a grammar bitmask (conservative: stay on
/// the sync path for constrained sequences).
pub fn async_launch_eligible(
    env: &AsyncEnvEligibility,
    has_markov: bool,
    grammar_masked: bool,
) -> bool {
    !has_markov
        && !grammar_masked
        // DeepLoop uses GPU-side commit-all (no inter-pass host D2H) so
        // denoise_steps > 1 is async-eligible when deeploop is active.
        && (env.denoise_steps <= 1 || env.deeploop)
        && !env.margin_gate_on
        && !env.adaptive_gamma
        && !env.tree_method
        && !env.portfolio
        && !env.cfg_jf
        && !env.kprofile
        && !env.debug_dump
        && !env.retrieval
        && !env.pld
        && !env.recycle
}

/// Placeholder chain returned by an async launch. Length == γ_eff so the
/// scheduler routes exactly as the sync path would (`len() >= 4` → K=γ+1
/// DFlash verify); values are the MASK token so that if a bug ever lets a
/// placeholder reach the verifier, every row mismatches the target argmax
/// and the step degrades to bonus-only — lossless by construction.
pub fn placeholder_drafts(gamma_eff: usize, mask_id: u32) -> Vec<u32> {
    vec![mask_id; gamma_eff]
}

/// Metadata for ATLAS_DFLASH_FREE_SLOTS async tree building.
/// Stored in `AsyncInflight.ddtree` when FREE_SLOTS >= 1 was active at
/// launch. At collect time, the scheduler D2Hs the top-K results (already
/// computed on the propose stream) and builds the tree payload on the CPU.
#[derive(Debug, Clone)]
pub struct AsyncDDTreeMeta {
    /// k used in `enqueue_topk_on_stream` (always 2 for FREE_SLOTS).
    pub top_k: usize,
    /// Number of sibling branches to place (= ATLAS_DFLASH_FREE_SLOTS).
    pub free_slots: usize,
    /// Top1−Top2 margin threshold for cliff detection (ATLAS_DFLASH_BRANCH_MARGIN).
    pub margin_thresh: f32,
    /// Post-cliff continuation tail length (ATLAS_DFLASH_FREE_SLOTS_TAIL).
    pub tail_len: usize,
    /// Immutable outer node cap for `build_free_slots_payload`.
    pub max_nodes: usize,
}

/// One in-flight async propose (at most one — the head has a single scratch
/// buffer set).
#[derive(Debug, Clone)]
pub struct AsyncInflight {
    /// Identity of the owning `DflashProposerState` (stable Box address for
    /// the sequence's lifetime). Matched at collect so a different sequence
    /// never consumes another's drafts.
    pub owner: usize,
    /// Exact public/trained/physical limits resolved by the launch call.
    pub budget: DflashDraftBudget,
    /// Stream the drafter kernels were enqueued on.
    pub stream: u64,
    /// When Some, a top-K GPU kernel was also enqueued on `stream` and the
    /// collect phase should D2H the results and build a tree payload.
    pub ddtree: Option<AsyncDDTreeMeta>,
}

/// Per-sequence materialized output from one synchronized batched async launch.
#[derive(Debug, Clone)]
pub struct AsyncBatchedSeqOutput {
    pub drafts: Vec<u32>,
    pub probs: Vec<f32>,
    pub rejection_q: Option<(usize, u64, usize)>,
}

/// One in-flight asynchronous cross-sequence batched (`C >= 2`) DFlash2 launch.
///
/// Uses alternating double-buffered device output slots (`Primary` / `Secondary`)
/// so consecutive launches never alias `draft_tokens_dev` or `batch_out_probs`.
/// Collection synchronizes `stream` and D2Hs all `B * gamma_eff` outputs ONCE on
/// the first sequence in `owners` to collect, caching the remaining sequences'
/// outputs for `O(1)` host-only handoff.
#[derive(Debug, Clone)]
pub struct AsyncBatchedInflight {
    pub slot: AsyncScratchSlot,
    pub owners: Vec<usize>,
    pub budget: DflashDraftBudget,
    pub gamma_eff: usize,
    pub stream: u64,
    pub writes_out_probs: bool,
    pub rejection_mode: bool,
    pub valid_q_records: Vec<Option<(usize, u64, usize)>>,
    pub materialized: Option<Vec<Option<AsyncBatchedSeqOutput>>>,
}

/// Metadata for an early-started Stage-1 (`fc` + `hidden_norm`) projection
/// launched on `pstream` after the target's final capture layer completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EarlyStage1Staged {
    pub owner: usize,
    pub start_pos: usize,
    pub num_rows: usize,
    pub stream: u64,
}

/// Stable identity for a proposer state (Box contents don't move).
pub fn dstate_id(dstate: &DflashProposerState) -> usize {
    dstate as *const DflashProposerState as usize
}

fn validate_async_collection(
    launched: DflashDraftBudget,
    current: Option<DflashDraftBudget>,
) -> Result<()> {
    if current != Some(launched) {
        anyhow::bail!("DFlash async budget changed before collection")
    }
    launched.validate_flat_width(launched.flat)
}

// ── Telemetry (fire / collect / discard counters, logged periodically) ──
static ASYNC_FIRES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ASYNC_COLLECTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ASYNC_DISCARDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn log_telemetry() {
    let f = ASYNC_FIRES.load(std::sync::atomic::Ordering::Relaxed);
    if f > 0 && f.is_multiple_of(256) {
        tracing::info!(
            "DFLASH_ASYNC telemetry: fires={f} collects={} discards={}",
            ASYNC_COLLECTS.load(std::sync::atomic::Ordering::Relaxed),
            ASYNC_DISCARDS.load(std::sync::atomic::Ordering::Relaxed),
        );
    }
}

impl BlockDiffusionDraftHead {
    /// Lazily-created dedicated propose stream (non-blocking). `0` means
    /// creation failed → async permanently disabled for this process.
    pub(super) fn propose_stream_lazy(&self, gpu: &dyn GpuBackend) -> u64 {
        *self
            .async_propose_stream
            .get_or_init(|| match (gpu.create_stream(), gpu.create_event()) {
                (Ok(s), Ok(e)) if s != 0 && e != 0 => {
                    self.async_order_event
                        .store(e, std::sync::atomic::Ordering::Release);
                    tracing::info!("DFLASH_ASYNC: created propose stream {s:#x} + order event");
                    s
                }
                (s, e) => {
                    tracing::warn!(
                        "DFLASH_ASYNC: stream/event creation failed (stream={s:?} event={e:?}) — \
                         async propose disabled"
                    );
                    0
                }
            })
    }

    /// Clear any unused `ATLAS_DFLASH_FUSED=1` pre-commit armed event so a host
    /// early-return or sync fallback never leaks a stale event into the next step.
    #[inline]
    pub(super) fn disarm_fused_overlap_event(&self) {
        self.fused_event_armed
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Atomically select the next alternating output scratch slot for async
    /// batched (`C >= 2`) proposals (`Primary -> Secondary -> Primary`).
    #[inline]
    pub(super) fn next_async_batch_slot(&self) -> AsyncScratchSlot {
        let prev = self
            .async_batch_slot_toggle
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        AsyncScratchSlot::from_u8(prev)
    }

    /// ATLAS_DFLASH_FUSED=1: record the propose-ordering CUDA event immediately
    /// after verify returns, BEFORE commit kernels are enqueued on the default
    /// stream. The drafter reads only `dflash_hidden_save` (populated by verify)
    /// and `ctx_hidden_acc` (per-sequence, not written by commit). Commit writes
    /// h_state and KV cache — disjoint from everything the drafter touches.
    /// Recording the event here lets the propose stream start while commit
    /// (~10ms SSM h_state copy + KV reshape) is still running on the default
    /// stream, instead of waiting for it.
    ///
    /// `try_launch_async_propose` detects the armed flag, skips re-recording,
    /// waits on `async_order_event` on `pstream`, and executes
    /// `append_pending_ctx_captures` ON `pstream` strictly AFTER that wait so
    /// there is zero cross-stream race on `ctx_hidden_acc`.
    /// No-op on ASYNC/FUSED flag off, or stream not yet created (falls back to
    /// record-at-launch on first step, fused on all subsequent ones).
    pub(crate) fn arm_propose_overlap(
        &self,
        gpu: &dyn GpuBackend,
        default_stream: u64,
    ) -> Result<()> {
        if !dflash_async_enabled() || !dflash_fused_enabled() {
            return Ok(());
        }
        let pstream = self.propose_stream_lazy(gpu);
        if pstream == 0 {
            return Ok(());
        }
        let ev = self
            .async_order_event
            .load(std::sync::atomic::Ordering::Acquire);
        if ev == 0 {
            return Ok(());
        }
        gpu.record_event(ev, default_stream)?;
        self.fused_event_armed
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Order `pstream` after `default_stream` (reusing an already-armed
    /// `ATLAS_DFLASH_FUSED=1` event when present), then execute any deferred
    /// `append_pending_ctx_captures` directly ON `pstream` strictly AFTER
    /// `stream_wait_event(pstream, ev)` so `pstream` never races with a
    /// default-stream D2D append into `ctx_hidden_acc`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn order_pstream_and_append_pending_ctx(
        &self,
        gpu: &dyn GpuBackend,
        default_stream: u64,
        pstream: u64,
        pending_ctx_base: Option<DevicePtr>,
        position: usize,
        dstate: &mut DflashProposerState,
    ) -> Result<(usize, usize)> {
        let ev = self
            .async_order_event
            .load(std::sync::atomic::Ordering::Acquire);
        let already_armed = self
            .fused_event_armed
            .swap(false, std::sync::atomic::Ordering::AcqRel);
        if !already_armed {
            gpu.record_event(ev, default_stream)?;
        }
        gpu.stream_wait_event(pstream, ev)?;

        let ctx_len_before = dstate.ctx_len;
        let num_append = if let Some(base) = pending_ctx_base {
            let last_num_accepted = dstate.last_num_accepted;
            let appended = match dstate.append_pending_ctx_captures(
                gpu,
                base,
                position,
                self.physical_verify_k,
                pstream,
            ) {
                Ok(n) => n,
                Err(error) => {
                    let _ = gpu.synchronize(pstream);
                    return Err(error);
                }
            };
            tracing::debug!(
                "DFlash async propose append (pstream): last_num_accepted={} num_append={} \
                 first_pos={} ctx_len_before={}",
                last_num_accepted,
                appended,
                ctx_len_before,
                ctx_len_before,
            );
            appended
        } else {
            0
        };
        Ok((ctx_len_before, num_append))
    }

    /// Order `pstream` after `default_stream` for a cross-sequence batched
    /// async DFlash2 launch. Always records `async_order_event` on
    /// `default_stream` (disarming any fused flag) because Phase-1 batched
    /// capture retirement wrote `ctx_hidden_acc` on `default_stream`.
    pub(super) fn order_pstream_for_batched_launch(
        &self,
        gpu: &dyn GpuBackend,
        default_stream: u64,
        pstream: u64,
    ) -> Result<()> {
        self.disarm_fused_overlap_event();
        let ev = self
            .async_order_event
            .load(std::sync::atomic::Ordering::Acquire);
        if ev == 0 {
            anyhow::bail!("DFLASH_ASYNC: batched launch missing async_order_event");
        }
        gpu.record_event(ev, default_stream)?;
        gpu.stream_wait_event(pstream, ev)?;
        Ok(())
    }

    /// Launch Stage-1 `fc` + `hidden_norm` early on `pstream` after waiting on
    /// `capture_done_event` (recorded inside target verify right after Layer 61
    /// on 27B or Layer 45 on Flash-Next), writing into the isolated staging slab
    /// `scratch.early_stage1_fc` without touching `dstate`'s live ring caches.
    ///
    /// Lossless (0-ULP): post-verify commits ONLY the accepted prefix
    /// `[0..num_accepted + 1]`; unaccepted or aborted rows are silently discarded.
    pub(crate) fn launch_early_stage1_from_capture_event(
        &self,
        gpu: &dyn GpuBackend,
        dflash_hidden_save: DevicePtr,
        num_rows: usize,
        start_pos: usize,
        capture_done_event: u64,
        dstate: &mut DflashProposerState,
    ) -> Result<()> {
        use crate::layers::ops;

        if !dflash_early_stage1_enabled()
            || self.scratch.early_stage1_fc.is_null()
            || dflash_hidden_save.is_null()
            || capture_done_event == 0
            || num_rows == 0
            || num_rows > self.physical_verify_k
            || dstate.ctx_fc_cache.is_null()
        {
            return Ok(());
        }
        let policy = self
            .proposal_policy
            .get_or_init(super::proposal_policy::ProposalPolicy::from_env);
        if policy.fc_layernorm
            || policy.debug_dump
            || policy.debug_dump_full
            || policy.dump_all_layers
            || policy.force_no_ctx
            || policy.zero_late > 0
            || policy.hf_override.is_some()
            || policy.force_pattern
            || policy.skip_decode_append
        {
            return Ok(());
        }
        self.resolve_async_inflight_impl(gpu, None)?;

        let pstream = self.propose_stream_lazy(gpu);
        if pstream == 0 {
            return Ok(());
        }
        let target_hidden_dim = self.target_layer_ids.len() * self.target_hidden_size;
        if target_hidden_dim == 0 {
            return Ok(());
        }
        let h = self.hidden_size as u32;
        let bf16 = 2usize;
        let ctx_slot_bytes = target_hidden_dim * bf16;
        let fc_dst = self.scratch.early_stage1_fc;

        gpu.stream_wait_event(pstream, capture_done_event)?;

        let launch_res: Result<()> = (|| {
            match (self.quant, self.fc_nvfp4.as_ref(), self.fc_fp8.as_ref()) {
                (super::DflashQuantization::Nvfp4, Some(fc_q), _) => {
                    self.nvfp4_project_coalesced_or_gemm(
                        gpu,
                        dflash_hidden_save,
                        fc_q,
                        fc_dst,
                        num_rows as u32,
                        h,
                        target_hidden_dim as u32,
                        pstream,
                    )?;
                }
                (super::DflashQuantization::Fp8, _, Some(fc_q)) => {
                    if num_rows == 1 && self.kernels.dense_gemv_fp8w.0 != 0 {
                        ops::dense_gemv_fp8w(
                            gpu,
                            self.kernels.dense_gemv_fp8w,
                            dflash_hidden_save,
                            fc_q,
                            fc_dst,
                            h,
                            target_hidden_dim as u32,
                            pstream,
                        )?;
                    } else if self.kernels.fp8_gemm_dflash.0 != 0 {
                        ops::fp8_gemm_dflash_rowscale(
                            gpu,
                            self.kernels.fp8_gemm_dflash,
                            dflash_hidden_save,
                            fc_q,
                            fc_dst,
                            num_rows as u32,
                            h,
                            target_hidden_dim as u32,
                            pstream,
                        )?;
                    } else {
                        for r in 0..num_rows {
                            ops::dense_gemv_fp8w(
                                gpu,
                                self.kernels.dense_gemv_fp8w,
                                dflash_hidden_save.offset(r * ctx_slot_bytes),
                                fc_q,
                                fc_dst.offset(r * self.hidden_size * bf16),
                                h,
                                target_hidden_dim as u32,
                                pstream,
                            )?;
                        }
                    }
                }
                _ => {
                    for r in 0..num_rows {
                        ops::dense_gemv(
                            gpu,
                            self.kernels.dense_gemv,
                            dflash_hidden_save.offset(r * ctx_slot_bytes),
                            &self.fc,
                            fc_dst.offset(r * self.hidden_size * bf16),
                            h,
                            target_hidden_dim as u32,
                            pstream,
                        )?;
                    }
                }
            }
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                fc_dst,
                &self.hidden_norm,
                fc_dst,
                num_rows as u32,
                h,
                self.rms_norm_eps,
                pstream,
            )?;
            Ok(())
        })();

        if let Err(error) = launch_res {
            let _ = gpu.synchronize(pstream);
            return Err(error);
        }

        *self.early_stage1_staged.lock() = Some(EarlyStage1Staged {
            owner: dstate_id(dstate),
            start_pos,
            num_rows,
            stream: pstream,
        });
        static EARLY_STAGE1_LOGGED: std::sync::Once = std::sync::Once::new();
        EARLY_STAGE1_LOGGED.call_once(|| {
            tracing::info!(
                num_rows,
                start_pos,
                "DFLASH_EARLY_STAGE1_ENGAGED (overlap Stage-1 fc+hidden_norm on pstream)"
            );
        });
        Ok(())
    }

    /// Commit ONLY the accepted prefix `[0..num_append)` from an early-started
    /// Stage-1 `fc` + `hidden_norm` staging buffer into `dstate.ctx_fc_cache`
    /// so `forward_block` Step 0 skips recomputing `fc` + `hidden_norm` for
    /// those rows (0-ULP bit-exact). Returns `Ok(true)` when committed.
    pub(super) fn try_commit_early_stage1_fc(
        &self,
        gpu: &dyn GpuBackend,
        dstate: &mut DflashProposerState,
        ctx_len_before: usize,
        num_append: usize,
        exec_stream: u64,
    ) -> Result<bool> {
        let Some(staged) = self.early_stage1_staged.lock().take() else {
            return Ok(false);
        };
        let matches = staged.owner == dstate_id(dstate)
            && staged.start_pos == ctx_len_before
            && num_append > 0
            && num_append <= staged.num_rows
            && !self.scratch.early_stage1_fc.is_null()
            && !dstate.ctx_fc_cache.is_null();
        if !matches {
            if staged.stream != exec_stream {
                gpu.synchronize(staged.stream)?;
            }
            return Ok(false);
        }
        if staged.stream != exec_stream {
            gpu.synchronize(staged.stream)?;
        }
        let row_bytes = self.hidden_size * 2;
        let (new_fc_start, new_fc_end) = self.cache_write_range(
            gpu,
            self.scratch.early_stage1_fc,
            ctx_len_before,
            num_append,
            dstate.ctx_fc_cache,
            dstate.cache_fc_start,
            dstate.cache_fc_end,
            self.ctx_window,
            row_bytes,
            exec_stream,
        )?;
        dstate.cache_fc_start = new_fc_start;
        dstate.cache_fc_end = new_fc_end;
        Ok(true)
    }

    /// Register a newly enqueued cross-sequence batched (`C >= 2`) async launch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn register_async_batched_inflight(
        &self,
        slot: AsyncScratchSlot,
        owners: Vec<usize>,
        budget: DflashDraftBudget,
        gamma_eff: usize,
        stream: u64,
        writes_out_probs: bool,
        rejection_mode: bool,
        valid_q_records: Vec<Option<(usize, u64, usize)>>,
    ) {
        *self.async_batched_inflight.lock() = Some(AsyncBatchedInflight {
            slot,
            owners,
            budget,
            gamma_eff,
            stream,
            writes_out_probs,
            rejection_mode,
            valid_q_records,
            materialized: None,
        });
        ASYNC_FIRES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        log_telemetry();
    }

    /// Resolve (sync + discard) any in-flight async propose (single-seq,
    /// cross-seq batched, or early-started Stage-1). Called before any new
    /// launch touches shared scratch, and from `free_sequence` before the
    /// per-seq ctx buffers are freed. Also clears an orphaned placeholder
    /// flag on `dstate` when provided.
    pub(crate) fn resolve_async_inflight_impl(
        &self,
        gpu: &dyn GpuBackend,
        mut dstate: Option<&mut DflashProposerState>,
    ) -> Result<()> {
        let taken_single = self.async_inflight.lock().take();
        let taken_batch = self.async_batched_inflight.lock().take();
        let taken_early = self.early_stage1_staged.lock().take();
        let mut synced_stream: Option<u64> = None;

        let mut sync_once = |stream: u64| -> Result<()> {
            if synced_stream == Some(stream) {
                return Ok(());
            }
            gpu.synchronize(stream)?;
            synced_stream = Some(stream);
            Ok(())
        };

        if let Some(inflight) = taken_single {
            if let Err(error) = sync_once(inflight.stream) {
                if let Some(ds) = dstate.as_deref_mut() {
                    clear_proposal_outputs(ds);
                }
                return Err(error);
            }
            ASYNC_DISCARDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!(
                "DFLASH_ASYNC: resolved stale in-flight propose (owner={:#x})",
                inflight.owner
            );
        }
        if let Some(batch) = taken_batch
            && batch.materialized.is_none()
        {
            if let Err(error) = sync_once(batch.stream) {
                if let Some(ds) = dstate.as_deref_mut() {
                    clear_proposal_outputs(ds);
                }
                return Err(error);
            }
            ASYNC_DISCARDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!(
                "DFLASH_ASYNC: resolved stale batched in-flight propose (slot={:?} owners={})",
                batch.slot,
                batch.owners.len(),
            );
        }
        if let Some(early) = taken_early
            && let Err(error) = sync_once(early.stream)
        {
            if let Some(ds) = dstate.as_deref_mut() {
                clear_proposal_outputs(ds);
            }
            return Err(error);
        }
        if let Some(ds) = dstate {
            ds.async_placeholder = false;
        }
        Ok(())
    }

    /// Try to launch the drafter forward asynchronously. Returns
    /// `Ok(Some(placeholder))` when the async path fired (drafts collected
    /// next step), `Ok(None)` when the caller must run the sync path.
    ///
    /// Ordering: records an event on `default_stream` (all prior writes the
    /// drafter reads — ctx-append D2Ds, verify captures — are enqueued
    /// there) and makes the propose stream wait on it, then enqueues the
    /// whole `forward_block` on the propose stream with the final
    /// synchronize + drafts D2H deferred.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_launch_async_propose(
        &self,
        last_token: u32,
        position: usize,
        budget: DflashDraftBudget,
        ctx: &crate::layer::ForwardContext,
        default_stream: u64,
        grammar_masked: bool,
        pending_ctx_base: Option<DevicePtr>,
        dstate: &mut DflashProposerState,
    ) -> Result<Option<Vec<u32>>> {
        budget.validate_flat_width(budget.flat)?;
        if !dflash_async_enabled() {
            return Ok(None);
        }
        // Sampled q has three paired D2H outputs and a drain-once scheduler
        // frame. Keep it on the synchronous producer boundary until async
        // collection owns that complete tuple rather than only token IDs.
        if self.draft_sample_method
            == crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic
        {
            return Ok(None);
        }
        let env = AsyncEnvEligibility::cached();
        if !async_launch_eligible(env, self.markov.is_some(), grammar_masked) {
            static INELIGIBLE_DBG: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !INELIGIBLE_DBG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(
                    "DFLASH_ASYNC: launch ineligible (env={env:?} markov={} grammar_masked={}) — \
                     sync propose path",
                    self.markov.is_some(),
                    grammar_masked,
                );
            }
            return Ok(None);
        }
        let gpu = ctx.gpu;
        let pstream = self.propose_stream_lazy(gpu);
        if pstream == 0 {
            return Ok(None);
        }
        // Preserve any matching early-stage1 handle across stale-inflight cleanup.
        let matching_early = {
            let mut early_guard = self.early_stage1_staged.lock();
            if early_guard
                .as_ref()
                .is_some_and(|e| e.owner == dstate_id(dstate) && e.stream == pstream)
            {
                early_guard.take()
            } else {
                None
            }
        };
        // Shared-scratch discipline: at most one in-flight propose.
        self.resolve_async_inflight_impl(gpu, None)?;
        if let Some(early) = matching_early {
            *self.early_stage1_staged.lock() = Some(early);
        }

        // GPU-side ordering + race-free ctx capture append:
        //   1. If `ATLAS_DFLASH_FUSED=1` already recorded `async_order_event`
        //      pre-commit, consume it; otherwise record now on `default_stream`.
        //   2. `stream_wait_event(pstream, ev)` ensures `dflash_hidden_save`
        //      writes from verify are visible on `pstream`.
        //   3. Execute `append_pending_ctx_captures` ON `pstream` strictly
        //      AFTER the event wait so `forward_block` on `pstream` NEVER races
        //      with a default-stream append into `ctx_hidden_acc`.
        let (ctx_len_before, num_append) = self.order_pstream_and_append_pending_ctx(
            gpu,
            default_stream,
            pstream,
            pending_ctx_base,
            position,
            dstate,
        )?;
        // Commit accepted prefix `[0..num_append)` from any early-started Stage-1
        // `fc` + `hidden_norm` directly on `pstream` (already stream-ordered!).
        if let Err(error) =
            self.try_commit_early_stage1_fc(gpu, dstate, ctx_len_before, num_append, pstream)
        {
            let _ = gpu.synchronize(pstream);
            return Err(error);
        }

        // Enqueue the drafter forward with the final sync + D2H deferred.
        // On a mid-enqueue failure, drain the propose stream BEFORE
        // returning: no inflight handle exists yet, and the caller's sync
        // fallback would otherwise rewrite the shared scratch while the
        // partially-enqueued kernels are still running.
        if let Err(e) = self.forward_block(
            last_token,
            position,
            budget.flat,
            ctx,
            pstream,
            dstate,
            true,
        ) {
            if let Err(sync_error) = gpu.synchronize(pstream) {
                anyhow::bail!(
                    "DFlash async enqueue failed ({e:#}) and stream drain failed ({sync_error:#})"
                )
            }
            return Err(e);
        }

        // ATLAS_DFLASH_FREE_SLOTS async DDTree path: enqueue top-K kernel on
        // pstream so collect_async_drafts_impl can D2H and build the tree
        // payload without a synchronous stream-sync+D2H during propose.
        // The forward_block logits (scratch.logits) are already enqueued on
        // pstream before this point, so ordering is guaranteed.
        let ddtree_meta = build_async_ddtree_meta(budget).and_then(|meta| {
            match self.enqueue_topk_on_stream(gpu, pstream, budget.flat, meta.top_k) {
                Ok(()) => Some(meta),
                Err(e) => {
                    tracing::warn!("DFLASH_ASYNC: top-K enqueue failed ({e:#}); no tree payload");
                    None
                }
            }
        });

        *self.async_inflight.lock() = Some(AsyncInflight {
            owner: dstate_id(dstate),
            budget,
            stream: pstream,
            ddtree: ddtree_meta,
        });
        dstate.async_placeholder = true;
        dstate.last_num_drafted = budget.flat;
        dstate.first_propose_done = true;
        dstate.pending_tree_payload = None;
        // `mtp_step.rs` already drained the previous step's collected tree
        // payload before invoking propose, so clearing `pending_tree_payload`
        // here prevents the post-propose sync drain from re-applying a stale tree.
        ASYNC_FIRES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        log_telemetry();
        Ok(Some(placeholder_drafts(budget.flat, self.mask_token_id)))
    }

    /// Collect the drafts of a previously-launched async propose (single-seq OR
    /// cross-seq batched) for this sequence.
    ///
    /// For batched (`C >= 2`) async launches, the first owner sequence to
    /// collect performs ONE stream synchronization + ONE D2H readback from the
    /// double-buffered slot (`Primary` or `Secondary`) and caches the cohort's
    /// outputs; remaining cohort owners pop their chain in `O(1)` with zero
    /// additional GPU syncs.
    ///
    /// Returns:
    ///   * `Ok(None)` — nothing pending for this sequence (normal sync path).
    ///   * `Ok(Some(drafts))` — real drafts; caller replaces the placeholder.
    ///   * `Ok(Some(vec![]))` — the placeholder was orphaned (handle lost to
    ///     a stale resolve); caller falls back to bootstrap decode. Lossless.
    pub(crate) fn collect_async_drafts_impl(
        &self,
        gpu: &dyn GpuBackend,
        dstate: &mut DflashProposerState,
    ) -> Result<Option<Vec<u32>>> {
        if let Some(res) = self.try_collect_async_batched_drafts(gpu, dstate)? {
            return Ok(Some(res));
        }

        let mut guard = self.async_inflight.lock();
        let matches = guard
            .as_ref()
            .is_some_and(|inf| inf.owner == dstate_id(dstate));
        if !matches {
            // A foreign in-flight handle (owner died / rerouted): resolve it
            // here so shared scratch is quiescent before this sequence's
            // verify/decode work touches the GPU.
            if let Some(stale) = guard.take() {
                drop(guard);
                if let Err(error) = gpu.synchronize(stale.stream) {
                    clear_proposal_outputs(dstate);
                    return Err(error);
                }
                ASYNC_DISCARDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if dstate.async_placeholder {
                tracing::warn!("DFLASH_ASYNC: orphaned placeholder — degrading to bootstrap");
                clear_proposal_outputs(dstate);
                return Ok(Some(Vec::new()));
            }
            return Ok(None);
        }
        let inflight = guard.take().expect("checked above");
        drop(guard);

        // Single stream sync covers both forward_block and (if present) top-K.
        if let Err(error) = gpu.synchronize(inflight.stream) {
            clear_proposal_outputs(dstate);
            return Err(error);
        }
        if validate_async_collection(inflight.budget, dstate.draft_budget).is_err() {
            tracing::warn!("DFLASH_ASYNC: stale or invalid launch budget; degrading to bootstrap");
            clear_proposal_outputs(dstate);
            return Ok(Some(Vec::new()));
        }
        let host_bytes = inflight
            .budget
            .flat
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| anyhow::anyhow!("DFlash async draft output byte count overflow"))?;
        let host_buf = &mut dstate.draft_output_pinned.as_mut_slice()[..host_bytes];
        if let Err(error) = gpu.copy_d2h(self.scratch.draft_tokens_dev, host_buf) {
            clear_proposal_outputs(dstate);
            return Err(error);
        }
        let drafts = super::proposal_inputs::decode_token_ids(host_buf, inflight.budget.flat)?;

        // ATLAS_DFLASH_FREE_SLOTS: D2H the top-K and build the tree payload.
        // The stream is already synced above, so `collect_topk_d2h` reads
        // the completed GPU output without an additional sync.
        if let Some(ref meta) = inflight.ddtree
            && drafts.len() >= 3
        {
            match self.collect_topk_d2h(gpu, inflight.budget.flat, meta.top_k) {
                Ok((topk_tokens, topk_logits)) if topk_logits.len() >= 2 * drafts.len() => {
                    let n = drafts.len();
                    let margins: Result<Vec<f32>> = (0..n)
                        .map(|r| {
                            checked_topk_difference(topk_logits[2 * r], topk_logits[2 * r + 1])
                        })
                        .collect();
                    match margins {
                        Ok(margins) => {
                            let cliffs = super::ddtree::pick_free_slot_cliffs(
                                &margins,
                                meta.margin_thresh,
                                meta.free_slots,
                            );
                            if !cliffs.is_empty() {
                                let cliff_margins: Vec<f32> =
                                    cliffs.iter().map(|&d| margins[d - 1]).collect();
                                tracing::debug!(
                                    "DFLASH_ASYNC collect: cliffs={:?} margins={:?} thresh={:.2}",
                                    cliffs,
                                    cliff_margins,
                                    meta.margin_thresh,
                                );
                            }
                            let branches: Vec<super::ddtree::FreeSlotBranch> = cliffs
                                .iter()
                                .filter_map(|&cliff_depth| {
                                    let row = cliff_depth - 1;
                                    let fork_token = topk_tokens[2 * row + 1];
                                    if fork_token == drafts[row] {
                                        return None;
                                    }
                                    let tail: Vec<u32> = drafts[(row + 1).min(n)..]
                                        .iter()
                                        .take(meta.tail_len)
                                        .copied()
                                        .collect();
                                    Some(super::ddtree::FreeSlotBranch {
                                        cliff_depth,
                                        fork_token,
                                        tail,
                                    })
                                })
                                .collect();
                            if !branches.is_empty() {
                                dstate.pending_tree_payload =
                                    Some(super::ddtree::build_free_slots_payload(
                                        &drafts,
                                        &branches,
                                        meta.max_nodes,
                                    ));
                            }
                        }
                        Err(error) => {
                            tracing::warn!(
                                "DFLASH_ASYNC: invalid top-K margins ({error:#}); no tree payload"
                            );
                            dstate.pending_tree_payload = None;
                        }
                    }
                }
                Ok(_) => {
                    dstate.pending_tree_payload = None;
                }
                Err(e) => {
                    tracing::warn!("DFLASH_ASYNC: top-K D2H failed ({e:#}); no tree payload");
                    dstate.pending_tree_payload = None;
                }
            }
        }

        dstate.async_placeholder = false;
        ASYNC_COLLECTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Probabilistic proposals are ineligible for async launch, so a frame
        // reaching this collector is necessarily the legacy greedy contract.
        Ok(Some(finalize_proposal(
            dstate,
            inflight.budget,
            drafts,
            false,
        )))
    }

    /// Collect one sequence's drafts from an in-flight cross-sequence batched
    /// (`C >= 2`) async launch. Synchronizes and D2Hs the double-buffered slot
    /// ONCE for the entire cohort on the first owner to arrive.
    fn try_collect_async_batched_drafts(
        &self,
        gpu: &dyn GpuBackend,
        dstate: &mut DflashProposerState,
    ) -> Result<Option<Vec<u32>>> {
        let owner_id = dstate_id(dstate);
        let mut guard = self.async_batched_inflight.lock();
        let Some(pos) = guard
            .as_ref()
            .and_then(|batch| batch.owners.iter().position(|&id| id == owner_id))
        else {
            // Foreign batched inflight handle while a different sequence is
            // collecting: drain it only if its GPU work has not synchronized yet.
            if let Some(stale) = guard.take() {
                drop(guard);
                if stale.materialized.is_none() {
                    if let Err(error) = gpu.synchronize(stale.stream) {
                        clear_proposal_outputs(dstate);
                        return Err(error);
                    }
                    ASYNC_DISCARDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            return Ok(None);
        };

        // Materialize the entire cohort in one sync + one D2H if not yet done.
        let need_materialize = guard
            .as_ref()
            .is_some_and(|batch| batch.materialized.is_none());
        if need_materialize {
            let Some(batch_meta) = guard.take() else {
                return Ok(None);
            };
            drop(guard);

            let b = batch_meta.owners.len();
            let gamma_eff = batch_meta.gamma_eff;
            let m_out = b
                .checked_mul(gamma_eff)
                .ok_or_else(|| anyhow::anyhow!("DFlash batched async output count overflow"))?;
            let out_bytes = m_out
                .checked_mul(size_of::<u32>())
                .ok_or_else(|| anyhow::anyhow!("DFlash batched async byte count overflow"))?;

            if let Err(error) = gpu.synchronize(batch_meta.stream) {
                clear_proposal_outputs(dstate);
                return Err(error);
            }
            let tok_ptr = self.scratch.draft_tokens_dev_for_slot(batch_meta.slot);
            let mut tok_host_buf = vec![0u8; out_bytes];
            if let Err(error) = gpu.copy_d2h(tok_ptr, &mut tok_host_buf) {
                clear_proposal_outputs(dstate);
                return Err(error);
            }
            let flat_tokens = super::proposal_inputs::decode_token_ids(&tok_host_buf, m_out)?;
            let mut probs_host_buf = Vec::new();
            if batch_meta.writes_out_probs {
                let probs_ptr = self.scratch.batch_out_probs_for_slot(batch_meta.slot);
                let prob_bytes = m_out.checked_mul(size_of::<f32>()).ok_or_else(|| {
                    anyhow::anyhow!("DFlash batched async prob byte count overflow")
                })?;
                probs_host_buf.resize(prob_bytes, 0u8);
                if let Err(error) = gpu.copy_d2h(probs_ptr, &mut probs_host_buf) {
                    clear_proposal_outputs(dstate);
                    return Err(error);
                }
            }

            let mut outputs: Vec<Option<AsyncBatchedSeqOutput>> = Vec::with_capacity(b);
            for s in 0..b {
                let start = s * gamma_eff;
                let drafts = flat_tokens[start..start + gamma_eff].to_vec();
                let probs = if batch_meta.writes_out_probs {
                    let prob_slice = &probs_host_buf
                        [start * size_of::<f32>()..(start + gamma_eff) * size_of::<f32>()];
                    prob_slice
                        .chunks_exact(size_of::<f32>())
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                } else {
                    Vec::new()
                };
                let rejection_q = if batch_meta.rejection_mode {
                    batch_meta.valid_q_records.get(s).copied().flatten()
                } else {
                    None
                };
                outputs.push(Some(AsyncBatchedSeqOutput {
                    drafts,
                    probs,
                    rejection_q,
                }));
            }
            let mut materialized_batch = batch_meta;
            materialized_batch.materialized = Some(outputs);
            guard = self.async_batched_inflight.lock();
            *guard = Some(materialized_batch);
        }

        let Some((budget, seq_output, all_consumed)) = guard.as_mut().map(|batch| {
            let budget = batch.budget;
            let entry = batch
                .materialized
                .as_mut()
                .and_then(|slots| slots.get_mut(pos))
                .and_then(Option::take);
            let done = batch
                .materialized
                .as_ref()
                .is_none_or(|slots| slots.iter().all(Option::is_none));
            (budget, entry, done)
        }) else {
            return Ok(None);
        };
        if all_consumed {
            *guard = None;
        }
        drop(guard);

        let Some(output) = seq_output else {
            if dstate.async_placeholder {
                tracing::warn!(
                    "DFLASH_ASYNC: batched placeholder already consumed — degrading to bootstrap"
                );
                clear_proposal_outputs(dstate);
                return Ok(Some(Vec::new()));
            }
            return Ok(None);
        };
        if validate_async_collection(budget, dstate.draft_budget).is_err() {
            tracing::warn!(
                "DFLASH_ASYNC: stale or invalid batched launch budget; degrading to bootstrap"
            );
            clear_proposal_outputs(dstate);
            return Ok(Some(Vec::new()));
        }
        dstate.async_placeholder = false;
        dstate.pending_tree_payload = None;
        dstate.pending_sampled_proposal = None;
        dstate.active_sampling_request = None;
        dstate.pending_draft_probs_gamma = 0;
        if let Some(rec) = output.rejection_q {
            dstate.set_last_rejection_q(Some(rec));
        } else {
            dstate.invalidate_rejection_q();
        }
        dstate.last_draft_probs.clear();
        dstate.last_draft_probs.extend_from_slice(&output.probs);
        ASYNC_COLLECTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Some(finalize_proposal(
            dstate,
            budget,
            output.drafts,
            false,
        )))
    }
}

/// Build `AsyncDDTreeMeta` from env vars when ATLAS_DFLASH_FREE_SLOTS >= 1.
/// Returns `None` when FREE_SLOTS is 0 or unset (no tree payload needed).
fn build_async_ddtree_meta(budget: DflashDraftBudget) -> Option<AsyncDDTreeMeta> {
    let free_slots: usize = std::env::var("ATLAS_DFLASH_FREE_SLOTS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    if free_slots == 0 {
        return None;
    }
    Some(AsyncDDTreeMeta {
        top_k: 2,
        free_slots,
        margin_thresh: std::env::var("ATLAS_DFLASH_BRANCH_MARGIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.0),
        tail_len: std::env::var("ATLAS_DFLASH_FREE_SLOTS_TAIL")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(4),
        max_nodes: budget.tree_nodes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::DraftProposer;

    fn all_clear() -> AsyncEnvEligibility {
        AsyncEnvEligibility {
            denoise_steps: 1,
            margin_gate_on: false,
            adaptive_gamma: false,
            tree_method: false,
            portfolio: false,
            cfg_jf: false,
            kprofile: false,
            debug_dump: false,
            retrieval: false,
            pld: false,
            recycle: false,
            deeploop: false,
        }
    }

    #[test]
    fn eligible_on_plain_flat_config() {
        assert!(async_launch_eligible(&all_clear(), false, false));
    }

    #[test]
    fn markov_head_disqualifies() {
        assert!(!async_launch_eligible(&all_clear(), true, false));
    }

    #[test]
    fn grammar_mask_disqualifies() {
        assert!(!async_launch_eligible(&all_clear(), false, true));
    }

    #[test]
    fn denoise_multi_pass_disqualifies() {
        let env = AsyncEnvEligibility {
            denoise_steps: 2,
            ..all_clear()
        };
        assert!(!async_launch_eligible(&env, false, false));
    }

    #[test]
    fn margin_gate_disqualifies() {
        let env = AsyncEnvEligibility {
            margin_gate_on: true,
            ..all_clear()
        };
        assert!(!async_launch_eligible(&env, false, false));
    }

    #[test]
    fn host_draft_preempt_paths_disqualify() {
        // retrieval/SAM, PLD, and recycle all pre-empt the neural drafter
        // with host drafts and early-return before async collect — each must
        // force the sync path (SAM+ASYNC init death, variants.md 2026-07-18).
        for f in [
            AsyncEnvEligibility {
                retrieval: true,
                ..all_clear()
            },
            AsyncEnvEligibility {
                pld: true,
                ..all_clear()
            },
            AsyncEnvEligibility {
                recycle: true,
                ..all_clear()
            },
        ] {
            assert!(!async_launch_eligible(&f, false, false));
        }
    }

    #[test]
    fn tree_methods_disqualify() {
        for f in [
            AsyncEnvEligibility {
                tree_method: true,
                ..all_clear()
            },
            AsyncEnvEligibility {
                portfolio: true,
                ..all_clear()
            },
        ] {
            assert!(!async_launch_eligible(&f, false, false));
        }
        let ddtree_env = AsyncEnvEligibility::from_lookup(
            |k| (k == "ATLAS_DFLASH_METHOD").then(|| "ddtree".to_string()),
            false,
        );
        assert!(ddtree_env.tree_method);
        assert!(!async_launch_eligible(&ddtree_env, false, false));

        let free_slots_env = AsyncEnvEligibility::from_lookup(
            |k| (k == "ATLAS_DFLASH_FREE_SLOTS").then(|| "2".to_string()),
            false,
        );
        assert!(!free_slots_env.tree_method);
        assert!(async_launch_eligible(&free_slots_env, false, false));
    }

    #[test]
    fn host_postprocessing_flags_disqualify() {
        for f in [
            AsyncEnvEligibility {
                adaptive_gamma: true,
                ..all_clear()
            },
            AsyncEnvEligibility {
                cfg_jf: true,
                ..all_clear()
            },
            AsyncEnvEligibility {
                kprofile: true,
                ..all_clear()
            },
            AsyncEnvEligibility {
                debug_dump: true,
                ..all_clear()
            },
        ] {
            assert!(!async_launch_eligible(&f, false, false));
        }
    }

    #[test]
    fn async_collection_rejects_changed_or_zero_budget() {
        let launched = DflashDraftBudget::new(4, 15, 16).unwrap();
        validate_async_collection(launched, Some(launched)).unwrap();
        assert!(validate_async_collection(launched, None).is_err());
        assert!(
            validate_async_collection(launched, Some(DflashDraftBudget::new(3, 15, 16).unwrap()))
                .is_err()
        );
        let zero = DflashDraftBudget::new(0, 15, 16).unwrap();
        assert!(validate_async_collection(zero, Some(zero)).is_err());
    }

    #[test]
    fn placeholder_is_all_mask_and_gamma_wide() {
        let p = placeholder_drafts(16, 42);
        assert_eq!(p.len(), 16);
        assert!(p.iter().all(|&t| t == 42));
        // Degenerate γ still yields a routable chain (scheduler needs >= 4
        // for the DFlash verify path; smaller is legal and falls to K=2/3/4).
        assert_eq!(placeholder_drafts(1, 7), vec![7]);
        assert!(placeholder_drafts(0, 7).is_empty());
    }

    #[test]
    fn final_dflash_capture_layer_matches_27b_and_flash_next() {
        // Dense 27B: [5, 19, 33, 47, 61] -> Layer 61 triggers early Stage-1.
        let layers_27b = [5, 19, 33, 47, 61];
        for l in [0, 5, 19, 33, 47, 60, 62, 63] {
            assert!(!is_final_dflash_capture_layer(l, &layers_27b));
        }
        assert!(is_final_dflash_capture_layer(61, &layers_27b));

        // Flash-Next 48L: [5, 15, 25, 35, 45] -> Layer 45 triggers early Stage-1.
        let layers_fn = [5, 15, 25, 35, 45];
        for l in [0, 5, 15, 25, 35, 44, 46, 47] {
            assert!(!is_final_dflash_capture_layer(l, &layers_fn));
        }
        assert!(is_final_dflash_capture_layer(45, &layers_fn));

        // Unsorted and empty slices behave safely.
        assert!(is_final_dflash_capture_layer(45, &[45, 5, 25]));
        assert!(!is_final_dflash_capture_layer(0, &[]));

        // Env parser defaults OFF.
        assert!(!parse_dflash_early_stage1_enabled(None));
        assert!(!parse_dflash_early_stage1_enabled(Some("0")));
        assert!(parse_dflash_early_stage1_enabled(Some("1")));
        assert!(parse_dflash_early_stage1_enabled(Some(" 1 ")));
    }

    #[test]
    fn dense_27b_pcum_policy_preserves_c1_default_and_admits_c4_or_thinking_opt_in() {
        use super::super::{
            dflash_pcum_enabled_for_target_context, parse_dflash_dense_pcum_opt_in,
        };

        assert!(!parse_dflash_dense_pcum_opt_in(None));
        assert!(!parse_dflash_dense_pcum_opt_in(Some("0")));
        assert!(parse_dflash_dense_pcum_opt_in(Some("1")));

        // Default OFF on dense 27B: disabled for all cohorts and thinking.
        assert!(!dflash_pcum_enabled_for_target_context(
            Some(false),
            1,
            false,
            false,
            None,
        ));
        assert!(!dflash_pcum_enabled_for_target_context(
            Some(false),
            8,
            true,
            false,
            None,
        ));

        // Opt-in ATLAS_DFLASH_DENSE_PCUM=1: preserves C=1..3 non-thinking full-K,
        // engages at C >= 4 or inside <think>.
        assert!(!dflash_pcum_enabled_for_target_context(
            Some(false),
            1,
            false,
            true,
            None,
        ));
        assert!(!dflash_pcum_enabled_for_target_context(
            Some(false),
            3,
            false,
            true,
            None,
        ));
        assert!(dflash_pcum_enabled_for_target_context(
            Some(false),
            4,
            false,
            true,
            None,
        ));
        assert!(dflash_pcum_enabled_for_target_context(
            Some(false),
            1,
            true,
            true,
            None,
        ));

        // MoE targets remain enabled by default; explicit threshold always wins.
        assert!(dflash_pcum_enabled_for_target_context(
            Some(true),
            1,
            false,
            false,
            None,
        ));
        assert!(!dflash_pcum_enabled_for_target_context(
            Some(true),
            8,
            true,
            true,
            Some("0"),
        ));
        assert!(dflash_pcum_enabled_for_target_context(
            Some(false),
            1,
            false,
            false,
            Some("0.25"),
        ));
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum RecordedGpuOp {
        RecordEvent {
            event: u64,
            stream: u64,
            mode: spark_runtime::gpu::EventRecordMode,
        },
        StreamWaitEvent {
            stream: u64,
            event: u64,
        },
        CopyD2dAsync {
            src: u64,
            dst: u64,
            bytes: usize,
            stream: u64,
        },
        Sync {
            stream: u64,
        },
        CopyD2h {
            src: u64,
            bytes: usize,
        },
    }

    struct RecordingAsyncGpu {
        inner: spark_runtime::gpu::mock::MockGpuBackend,
        ops: std::sync::Mutex<Vec<RecordedGpuOp>>,
        next_stream: std::sync::atomic::AtomicU64,
        next_event: std::sync::atomic::AtomicU64,
    }

    impl RecordingAsyncGpu {
        fn new() -> Self {
            Self {
                inner: spark_runtime::gpu::mock::MockGpuBackend::new(),
                ops: std::sync::Mutex::new(Vec::new()),
                next_stream: std::sync::atomic::AtomicU64::new(0x100),
                next_event: std::sync::atomic::AtomicU64::new(0x200),
            }
        }

        fn take_ops(&self) -> Vec<RecordedGpuOp> {
            std::mem::take(&mut *self.ops.lock().unwrap())
        }
    }

    impl GpuBackend for RecordingAsyncGpu {
        fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
            self.inner.alloc(bytes)
        }
        fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
            self.inner.alloc_managed(bytes)
        }
        fn free(&self, ptr: DevicePtr) -> Result<()> {
            self.inner.free(ptr)
        }
        fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
            self.inner.copy_h2d(src, dst)
        }
        fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
            self.ops.lock().unwrap().push(RecordedGpuOp::CopyD2h {
                src: src.0,
                bytes: dst.len(),
            });
            self.inner.copy_d2h(src, dst)
        }
        fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
            self.inner.copy_d2d(src, dst, bytes)
        }
        fn copy_d2d_async(
            &self,
            src: DevicePtr,
            dst: DevicePtr,
            bytes: usize,
            stream: u64,
        ) -> Result<()> {
            self.ops.lock().unwrap().push(RecordedGpuOp::CopyD2dAsync {
                src: src.0,
                dst: dst.0,
                bytes,
                stream,
            });
            self.inner.copy_d2d_async(src, dst, bytes, stream)
        }
        fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
            self.inner.memset(ptr, value, bytes)
        }
        fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
            self.inner.memset_async(ptr, value, bytes, stream)
        }
        fn kernel(&self, module: &str, func: &str) -> Result<spark_runtime::gpu::KernelHandle> {
            self.inner.kernel(module, func)
        }
        fn launch(
            &self,
            kernel: spark_runtime::gpu::KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            shared_mem: u32,
            stream: u64,
            args: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            self.inner
                .launch(kernel, grid, block, shared_mem, stream, args)
        }
        fn default_stream(&self) -> u64 {
            1
        }
        fn create_stream(&self) -> Result<u64> {
            Ok(self
                .next_stream
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed))
        }
        fn create_event(&self) -> Result<u64> {
            Ok(self
                .next_event
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed))
        }
        fn record_event(&self, event: u64, stream: u64) -> Result<()> {
            self.ops.lock().unwrap().push(RecordedGpuOp::RecordEvent {
                event,
                stream,
                mode: spark_runtime::gpu::EventRecordMode::Default,
            });
            Ok(())
        }
        fn record_event_with_mode(
            &self,
            event: u64,
            stream: u64,
            mode: spark_runtime::gpu::EventRecordMode,
        ) -> Result<()> {
            self.ops.lock().unwrap().push(RecordedGpuOp::RecordEvent {
                event,
                stream,
                mode,
            });
            Ok(())
        }
        fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
            self.ops
                .lock()
                .unwrap()
                .push(RecordedGpuOp::StreamWaitEvent { stream, event });
            Ok(())
        }
        fn destroy_event(&self, _event: u64) -> Result<()> {
            Ok(())
        }
        fn synchronize(&self, stream: u64) -> Result<()> {
            self.ops
                .lock()
                .unwrap()
                .push(RecordedGpuOp::Sync { stream });
            Ok(())
        }
        fn total_memory(&self) -> Result<usize> {
            self.inner.total_memory()
        }
        fn free_memory(&self) -> Result<usize> {
            self.inner.free_memory()
        }
    }

    fn as_dstate_mut(
        state: &mut dyn crate::speculative::ProposerState,
    ) -> &mut DflashProposerState {
        state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .expect("DflashProposerState")
    }

    fn as_dstate_ref(state: &dyn crate::speculative::ProposerState) -> &DflashProposerState {
        state
            .as_any()
            .downcast_ref::<DflashProposerState>()
            .expect("DflashProposerState")
    }

    fn make_test_head_and_states(
        gpu: &RecordingAsyncGpu,
        count: usize,
    ) -> (
        Box<BlockDiffusionDraftHead>,
        Vec<Box<dyn crate::speculative::ProposerState>>,
        DevicePtr,
    ) {
        let mut head = super::super::seq_buf_pool_tests::test_draft_head(5);
        let save_buf = gpu.alloc(head.physical_verify_k * 5 * 32 * 2).unwrap();
        let mut owned_states = Vec::with_capacity(count);
        for _ in 0..count {
            owned_states.push(head.alloc_state(gpu).unwrap());
        }
        head.scratch.draft_tokens_dev = gpu.alloc(64 * 4).unwrap();
        head.scratch.draft_tokens_dev_alt = gpu.alloc(64 * 4).unwrap();
        head.scratch.batch_out_probs = gpu.alloc(64 * 4).unwrap();
        head.scratch.batch_out_probs_alt = gpu.alloc(64 * 4).unwrap();
        head.scratch.early_stage1_fc = gpu
            .alloc(head.physical_verify_k * head.hidden_size * 2)
            .unwrap();
        (head, owned_states, save_buf)
    }

    #[test]
    fn fused_async_appends_ctx_captures_on_pstream_strictly_after_event_wait() {
        let gpu = RecordingAsyncGpu::new();
        let (head, mut states, save_buf) = make_test_head_and_states(&gpu, 1);
        let dstate = as_dstate_mut(states[0].as_mut());
        dstate.last_num_accepted = 2; // expects 3 rows appended (anchor + 2 accepted)
        let default_stream = gpu.default_stream();
        let pstream = head.propose_stream_lazy(&gpu);
        let ev = head
            .async_order_event
            .load(std::sync::atomic::Ordering::Acquire);
        assert_ne!(pstream, 0);
        assert_ne!(pstream, default_stream);
        assert_ne!(ev, 0);

        // Simulate `ATLAS_DFLASH_FUSED=1` arming pre-commit on default_stream.
        gpu.record_event(ev, default_stream).unwrap();
        head.fused_event_armed
            .store(true, std::sync::atomic::Ordering::Release);

        let (before, appended) = head
            .order_pstream_and_append_pending_ctx(
                &gpu,
                default_stream,
                pstream,
                Some(save_buf),
                3,
                dstate,
            )
            .unwrap();
        assert_eq!(before, 0);
        assert_eq!(appended, 3);
        assert_eq!(dstate.ctx_len, 3);
        assert!(
            !head
                .fused_event_armed
                .load(std::sync::atomic::Ordering::Acquire)
        );

        let ops = gpu.take_ops();
        // Exact order: RecordEvent(default) -> StreamWaitEvent(pstream) -> CopyD2dAsync(pstream).
        // Zero second RecordEvent, zero CopyD2dAsync on default_stream!
        assert!(
            ops.len() >= 3,
            "expected record, wait, and d2d ops: {ops:?}"
        );
        assert_eq!(
            ops[0],
            RecordedGpuOp::RecordEvent {
                event: ev,
                stream: default_stream,
                mode: spark_runtime::gpu::EventRecordMode::Default,
            }
        );
        assert_eq!(
            ops[1],
            RecordedGpuOp::StreamWaitEvent {
                stream: pstream,
                event: ev,
            }
        );
        for op in &ops[2..] {
            match op {
                RecordedGpuOp::CopyD2dAsync { stream, .. } => {
                    assert_eq!(
                        *stream, pstream,
                        "ctx capture D2D must execute on pstream after StreamWaitEvent, not default_stream"
                    );
                }
                other => panic!("unexpected op after wait: {other:?}"),
            }
        }

        // Non-fused path (fused_event_armed == false) records event on default_stream,
        // waits on pstream, then appends on pstream.
        dstate.last_num_accepted = 1; // expects 2 more rows -> position 5
        let (before2, appended2) = head
            .order_pstream_and_append_pending_ctx(
                &gpu,
                default_stream,
                pstream,
                Some(save_buf),
                5,
                dstate,
            )
            .unwrap();
        assert_eq!(before2, 3);
        assert_eq!(appended2, 2);
        let ops2 = gpu.take_ops();
        assert_eq!(
            ops2[0],
            RecordedGpuOp::RecordEvent {
                event: ev,
                stream: default_stream,
                mode: spark_runtime::gpu::EventRecordMode::Default,
            }
        );
        assert_eq!(
            ops2[1],
            RecordedGpuOp::StreamWaitEvent {
                stream: pstream,
                event: ev,
            }
        );
        assert!(ops2[2..].iter().all(|op| matches!(
            op,
            RecordedGpuOp::CopyD2dAsync { stream, .. } if *stream == pstream
        )));

        // Cleanup pooled state buffers cleanly.
        head.release_rejection_resources(&gpu);
    }

    #[test]
    fn early_stage1_commits_only_accepted_prefix_and_discards_mismatched_owner() {
        let gpu = RecordingAsyncGpu::new();
        let (head, mut states, _save_buf) = make_test_head_and_states(&gpu, 2);
        let pstream = head.propose_stream_lazy(&gpu);
        let d0 = as_dstate_mut(states[0].as_mut());

        // Populate 4 rows in `early_stage1_fc` with distinct byte markers.
        let row_bytes = head.hidden_size * 2;
        let mut staged_bytes = vec![0u8; 4 * row_bytes];
        for r in 0..4 {
            staged_bytes[r * row_bytes..(r + 1) * row_bytes].fill((r + 1) as u8);
        }
        gpu.copy_h2d(&staged_bytes, head.scratch.early_stage1_fc)
            .unwrap();
        *head.early_stage1_staged.lock() = Some(EarlyStage1Staged {
            owner: dstate_id(d0),
            start_pos: 0,
            num_rows: 4,
            stream: pstream,
        });
        gpu.take_ops();

        // Only 2 of 4 verified rows were accepted -> commit ONLY [0..2) on pstream.
        let committed = head
            .try_commit_early_stage1_fc(&gpu, d0, 0, 2, pstream)
            .unwrap();
        assert!(committed);
        assert_eq!(d0.cache_fc_start, 0);
        assert_eq!(d0.cache_fc_end, 2);
        let ops = gpu.take_ops();
        assert_eq!(ops.len(), 1);
        assert!(matches!(
            ops[0],
            RecordedGpuOp::CopyD2dAsync { bytes, stream, .. }
                if bytes == 2 * row_bytes && stream == pstream
        ));
        let mut readback = vec![0u8; 4 * row_bytes];
        gpu.copy_d2h(d0.ctx_fc_cache, &mut readback).unwrap();
        assert!(readback[..row_bytes].iter().all(|&b| b == 1));
        assert!(readback[row_bytes..2 * row_bytes].iter().all(|&b| b == 2));
        // Rows 2..4 (rejected draft tokens) must NOT have been committed.
        assert!(readback[2 * row_bytes..].iter().all(|&b| b == 0));

        // Mismatched owner on a different stream syncs and discards without committing.
        let owner0 = dstate_id(as_dstate_ref(states[0].as_ref()));
        let d1 = as_dstate_mut(states[1].as_mut());
        *head.early_stage1_staged.lock() = Some(EarlyStage1Staged {
            owner: owner0,
            start_pos: 2,
            num_rows: 4,
            stream: pstream,
        });
        gpu.take_ops();
        let committed_wrong = head
            .try_commit_early_stage1_fc(&gpu, d1, 2, 2, gpu.default_stream())
            .unwrap();
        assert!(!committed_wrong);
        assert_eq!(d1.cache_fc_end, 0);
        assert_eq!(
            gpu.take_ops(),
            vec![RecordedGpuOp::Sync { stream: pstream }]
        );

        head.release_rejection_resources(&gpu);
    }

    #[test]
    fn batched_async_double_buffers_slots_and_collects_cohort_in_one_sync() {
        let gpu = RecordingAsyncGpu::new();
        let (head, mut states, _save_buf) = make_test_head_and_states(&gpu, 3);
        let pstream = head.propose_stream_lazy(&gpu);

        // Consecutive launches strictly alternate Primary -> Secondary -> Primary
        // with distinct device output pointers (zero aliasing).
        let s0 = head.next_async_batch_slot();
        let s1 = head.next_async_batch_slot();
        let s2 = head.next_async_batch_slot();
        assert_eq!(s0, AsyncScratchSlot::Primary);
        assert_eq!(s1, AsyncScratchSlot::Secondary);
        assert_eq!(s2, AsyncScratchSlot::Primary);
        assert_ne!(
            head.scratch
                .draft_tokens_dev_for_slot(AsyncScratchSlot::Primary),
            head.scratch
                .draft_tokens_dev_for_slot(AsyncScratchSlot::Secondary)
        );
        assert_ne!(
            head.scratch
                .batch_out_probs_for_slot(AsyncScratchSlot::Primary),
            head.scratch
                .batch_out_probs_for_slot(AsyncScratchSlot::Secondary)
        );

        // Populate Secondary slot device buffers for a 3-seq cohort (gamma_eff = 2).
        let gamma_eff = 2usize;
        let budget = DflashDraftBudget::new(gamma_eff, 7, 8).unwrap();
        let tokens: [u32; 6] = [11, 12, 21, 22, 31, 32];
        let probs: [f32; 6] = [0.91, 0.82, 0.73, 0.64, 0.55, 0.46];
        let tok_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        let prob_bytes: Vec<u8> = probs.iter().flat_map(|p| p.to_le_bytes()).collect();
        let sec_tok_ptr = head
            .scratch
            .draft_tokens_dev_for_slot(AsyncScratchSlot::Secondary);
        let sec_prob_ptr = head
            .scratch
            .batch_out_probs_for_slot(AsyncScratchSlot::Secondary);
        gpu.copy_h2d(&tok_bytes, sec_tok_ptr).unwrap();
        gpu.copy_h2d(&prob_bytes, sec_prob_ptr).unwrap();

        let mut owners = Vec::new();
        for st in &mut states {
            let ds = as_dstate_mut(st.as_mut());
            ds.draft_budget = Some(budget);
            ds.async_placeholder = true;
            owners.push(dstate_id(ds));
        }
        head.register_async_batched_inflight(
            AsyncScratchSlot::Secondary,
            owners,
            budget,
            gamma_eff,
            pstream,
            true,
            true,
            vec![Some((0, 10, 2)), Some((1, 20, 2)), Some((2, 30, 2))],
        );
        gpu.take_ops();

        // Sequence 0 triggers the ONLY stream sync + D2H from Secondary slot.
        let d0 = head
            .collect_async_drafts_impl(&gpu, as_dstate_mut(states[0].as_mut()))
            .unwrap();
        assert_eq!(d0, Some(vec![11, 12]));
        assert_eq!(
            as_dstate_ref(states[0].as_ref()).last_draft_probs,
            vec![0.91, 0.82]
        );
        assert_eq!(
            as_dstate_ref(states[0].as_ref()).last_rejection_q(),
            Some((0, 10, 2))
        );
        assert!(!as_dstate_ref(states[0].as_ref()).async_placeholder);
        let ops_after_0 = gpu.take_ops();
        assert_eq!(
            ops_after_0,
            vec![
                RecordedGpuOp::Sync { stream: pstream },
                RecordedGpuOp::CopyD2h {
                    src: sec_tok_ptr.0,
                    bytes: 24,
                },
                RecordedGpuOp::CopyD2h {
                    src: sec_prob_ptr.0,
                    bytes: 24,
                },
            ]
        );

        // Sequences 1 & 2 pop materialized outputs with ZERO extra GPU ops.
        let d1 = head
            .collect_async_drafts_impl(&gpu, as_dstate_mut(states[1].as_mut()))
            .unwrap();
        let d2 = head
            .collect_async_drafts_impl(&gpu, as_dstate_mut(states[2].as_mut()))
            .unwrap();
        assert_eq!(d1, Some(vec![21, 22]));
        assert_eq!(
            as_dstate_ref(states[1].as_ref()).last_draft_probs,
            vec![0.73, 0.64]
        );
        assert_eq!(
            as_dstate_ref(states[1].as_ref()).last_rejection_q(),
            Some((1, 20, 2))
        );
        assert_eq!(d2, Some(vec![31, 32]));
        assert_eq!(
            as_dstate_ref(states[2].as_ref()).last_draft_probs,
            vec![0.55, 0.46]
        );
        assert_eq!(
            as_dstate_ref(states[2].as_ref()).last_rejection_q(),
            Some((2, 30, 2))
        );
        assert!(gpu.take_ops().is_empty());
        assert!(head.async_batched_inflight.lock().is_none());

        // Stale uncollected batched inflight is cleanly drained by resolve_async_inflight_impl.
        as_dstate_mut(states[0].as_mut()).async_placeholder = true;
        let owner0 = dstate_id(as_dstate_ref(states[0].as_ref()));
        let owner1 = dstate_id(as_dstate_ref(states[1].as_ref()));
        head.register_async_batched_inflight(
            AsyncScratchSlot::Primary,
            vec![owner0, owner1],
            budget,
            gamma_eff,
            pstream,
            false,
            false,
            vec![None, None],
        );
        head.resolve_async_inflight_impl(&gpu, Some(as_dstate_mut(states[0].as_mut())))
            .unwrap();
        assert!(head.async_batched_inflight.lock().is_none());
        assert!(!as_dstate_ref(states[0].as_ref()).async_placeholder);
        assert_eq!(
            gpu.take_ops(),
            vec![RecordedGpuOp::Sync { stream: pstream }]
        );

        head.release_rejection_resources(&gpu);
    }
}
