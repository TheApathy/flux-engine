// SPDX-License-Identifier: AGPL-3.0-only

//! Scheduler: batched concurrent decode on a single GPU thread.
//!
//! Architecture:
//! - Receiver thread: blocks on request channel, pushes to pending queue,
//!   signals condvar (instantaneous wake, zero polling).
//! - Scheduler thread: prefills new requests sequentially, then runs
//!   batched decode steps via `model.decode_batch()` — weights loaded once
//!   per step for all active sequences.
//!
//! When idle (no active sequences): blocks on condvar (zero CPU).
//! When busy: drains pending queue (mutex lock) after each decode step.

// ── Submodules (split for ≤500 LoC files) ──────────────────────────────────
mod adaptive_depth;
mod admission;
mod cfg_jump_forward;
mod confidence;
pub mod continuation;
mod decode_logits_content;
mod decode_logits_seq;
mod decode_logits_step;
mod decode_step;
mod deferred_prefill_tail;
mod device_topk_route;
mod device_topk_spec;
mod emit_step;
mod exact_greedy_admission;
mod glm53_policy_driver;
mod helpers;
mod internal_spec_step;
pub(crate) mod kv_headroom;
// Run-scoped levers the dashboard toggles. Minimal port: the loop watchdog only.
pub mod levers;
mod lifecycle;
mod logits_row_receipt;
mod logprobs;
mod lookup_receipt;
mod mod_helpers;
mod mtp_gate;
mod mtp_step;
mod phase_continue_prefills;
mod phase_promote_prefills;
mod phase_start_prefills;
mod ple_queue_prefetch;
mod prefill_a_step;
#[cfg(test)]
mod prefill_a_step_tests;
mod prefill_b_step;
pub mod prefill_chunk_policy;
mod prefill_interleave;
mod prefill_lifecycle;
mod prefill_slo;
mod prefill_slo_cost;
mod prefill_stream_barrier;
mod proposal_lifecycle;
pub(crate) mod queue_gate;
mod qwen4_batch_route;
mod repetition;
pub(crate) mod retirement;
mod rollback;
mod sample_step;
mod sampled_dflash;
#[cfg(test)]
mod settled_frame_receipt_tests;
mod short_verify_frame;
#[cfg(test)]
mod single_tool_close_tests;
pub mod snapshot;
mod spec_accept_log;
mod spec_policy_accept;
mod spec_step;
mod spec_timing;
#[cfg(test)]
mod speculation_control_tests;
mod ssm_decode_ring;
mod stochastic_verify;
#[cfg(test)]
mod stochastic_verify_tests;
mod stochastic_verify_walk;
#[cfg(test)]
mod stochastic_verify_walk_tests;
mod stop_ids;
#[cfg(test)]
mod test_fixtures;
pub mod think_budget_nudge;
pub mod think_nudge;
mod think_spec_accept;
mod thinking_efficiency;
mod types;
mod verify_csk_step;
mod verify_csk_step_k2;
mod verify_dflash_batched_step;
mod verify_dflash_step;
mod verify_k2_step;
mod verify_k3_step;
mod verify_k4_step;
mod verify_qwen4_batched_step;
#[cfg(test)]
mod vision_request_reset_tests;

pub use cfg_jump_forward::{
    build_delim_table, build_forced_ids, cfg_jf_enabled, set_delim_table, set_forced_ids,
};
use confidence::*;
use decode_logits_content::*;
use decode_logits_seq::*;
use decode_logits_step::*;
use decode_step::*;
use deferred_prefill_tail::finish_deferred_prefill_tail;
pub use device_topk_route::device_topk_startup_receipt;
use emit_step::*;
pub use helpers::enable_loop_watchdog;
pub use helpers::set_boundary_token_mask;
pub use helpers::set_enable_catastrophic_loop_watchdog;
pub use helpers::set_enable_loop_watchdog;
pub use helpers::set_enable_thinking_loop_watchdog;
pub use helpers::set_honor_eos_inside_thinking;
pub use helpers::set_im_start_hard_stop;
pub use helpers::set_numeric_token_mask;
use helpers::*;
pub use helpers::{CONTENT_LOOP_PERIOD_MAX, CONTENT_LOOP_PERIOD_MIN};
pub use helpers::{WatchdogParams, resolve_max_inter_tool_prose, set_watchdog_params};
use internal_spec_step::*;
use lifecycle::*;
use logprobs::*;
use mod_helpers::*;
use mtp_gate::{ArmKind, ArmSpec, MtpGate};
use mtp_step::*;
use phase_continue_prefills::continue_in_progress_prefills;
use phase_start_prefills::start_new_requests;
use prefill_a_step::*;
use prefill_b_step::*;
use prefill_lifecycle::*;
use repetition::*;
use rollback::{RollbackOutcome, rollback_to_boundary};
use sample_step::*;
use sampled_dflash::*;
use short_verify_frame::ShortVerifyFrame;
use spec_accept_log::*;
use spec_policy_accept::*;
use spec_step::*;
use ssm_decode_ring::SsmDecodeRing;
use stop_ids::sequence_stop_ids;
use think_spec_accept::*;
pub use thinking_efficiency::{
    ADAPTIVE_PROBE_TOKENS as ADAPTIVE_PROBE_TOKENS_LOG, apply_think_logit_shaping,
    build_hesitation_ids, parse_config as parse_think_efficiency_config,
    set_think_efficiency_config, think_efficiency_config, top1_confidence,
};
use types::*;
use verify_csk_step::*;
use verify_csk_step_k2::*;
use verify_dflash_batched_step::*;
use verify_dflash_step::*;
use verify_k2_step::*;
use verify_k3_step::*;
use verify_k4_step::*;

// Re-exports threaded through `use super::*;` in sibling step files —
// keep these imports here even though `run` itself doesn't reference all
// of them directly (see scheduler/decode_step.rs etc.).
use anyhow::Result;
use parking_lot::{Condvar, Mutex};
use spark_model::traits::{EP_CMD_VERIFY_KGAMMA, EP_VERIFY_KGAMMA_ABORT, Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_spill::KvSpillManager;
use spark_runtime::sampler::{SamplingParams, sample_with_params_history};

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::api::{GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::ngram::NgramProposer;
use crate::scheduling_policy::SchedulingPolicy;

/// Build the speculation gate's arm set, or `None` to leave it disarmed.
///
/// Disarmed is the DEFAULT and must stay that way: `--mtp-gate` absent means
/// the champion configuration runs byte-for-byte as it did before the gate
/// existed, which is what makes an A/B against it meaningful.
///
/// `mode` is the resolved `--mtp-gate` value:
///   - `None`            → no gate (default)
///   - `Some("force")`   → no gate, and say so (diagnostic; verify keeps
///                          flowing even where the gate would measure it
///                          net-negative)
///   - `Some("auto")`    → arbitrate over every arm this build can offer
///   - `Some("dflash")`  → pin proposer arm 0
///   - `Some("mtp")`     → pin proposer arm 1
///
/// Optional extra arms, opt-in because neither is measured yet:
///   - `ATLAS_MTP_GATE_GAMMA_ARM=N` adds a γ-capped variant of arm 0. See
///     `speculative::set_dflash_gamma_override` for why a runtime cap is not
///     the same thing as serving with `--dflash-gamma N`.
///   - `ATLAS_MTP_GATE_SERIAL_ARM=1` adds a no-speculation arm. Measured floor
///     is ~13 tok/s on every task, i.e. dominated by both proposer arms on
///     this model — hence off by default.
fn build_speculation_gate(model: &dyn Model, use_mtp: bool, mode: Option<&str>) -> Option<MtpGate> {
    let mode = mode?;
    if !use_mtp {
        tracing::warn!("--mtp-gate {mode} ignored: speculation is not active for this run");
        return None;
    }
    if mode == "force" {
        tracing::warn!(
            "--mtp-gate force: speculation gate DISARMED (diagnostic; verify runs even \
             where the gate would measure it net-negative)"
        );
        return None;
    }

    let arms_available = model.proposer_arm_count();
    let primary_is_dflash = model.proposer_is_dflash();
    let primary_name = if primary_is_dflash { "dflash" } else { "mtp" };
    // The alternate arm is the in-checkpoint MTP head, which is measured
    // monotonically worse with more drafts on this model family (K=2 beats
    // K=4 on every benchmark task; K=8 falls below the no-speculation floor).
    // The run's `num_drafts` belongs to the DFlash arm — with a block-16
    // checkpoint it is the actual trained draft count γ=15 — so the MTP arm
    // gets its own, overridable for a
    // model whose head really does want a wider K.
    let alt_num_drafts = std::env::var("ATLAS_MTP_GATE_ALT_DRAFTS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(1);

    // Pinned arms are a one-arm "gate": no probing, no switching, but the
    // arm-selection plumbing still runs so `--mtp-gate mtp` is a real A/B
    // against `--mtp-gate dflash` on one binary and one set of weights.
    match mode {
        "dflash" => {
            return Some(MtpGate::new(vec![ArmSpec::spec(
                "pinned-dflash",
                mtp_gate::PROPOSER_ARM_PRIMARY,
                0,
                0,
            )]));
        }
        "mtp" => {
            if arms_available < 2 {
                tracing::warn!(
                    "--mtp-gate mtp: this build has only one proposer arm ({primary_name}); \
                     pass --dflash AND --speculative to build both. Pinning arm 0 instead."
                );
                return Some(MtpGate::new(vec![ArmSpec::spec(
                    "pinned-arm0",
                    mtp_gate::PROPOSER_ARM_PRIMARY,
                    0,
                    0,
                )]));
            }
            return Some(MtpGate::new(vec![ArmSpec::spec(
                "pinned-mtp",
                mtp_gate::PROPOSER_ARM_ALT,
                0,
                alt_num_drafts,
            )]));
        }
        "auto" => {}
        other => {
            tracing::error!("--mtp-gate {other}: unknown mode, gate DISARMED");
            return None;
        }
    }

    let mut arms = vec![ArmSpec::spec(
        if primary_is_dflash { "dflash" } else { "mtp" },
        mtp_gate::PROPOSER_ARM_PRIMARY,
        0,
        0,
    )];
    if arms_available >= 2 {
        arms.push(ArmSpec::spec(
            "mtp-alt",
            mtp_gate::PROPOSER_ARM_ALT,
            0,
            alt_num_drafts,
        ));
    }
    if primary_is_dflash
        && let Ok(v) = std::env::var("ATLAS_MTP_GATE_GAMMA_ARM")
        && let Ok(g) = v.trim().parse::<usize>()
        && g > 0
    {
        arms.push(ArmSpec::spec(
            "dflash-gamma-capped",
            mtp_gate::PROPOSER_ARM_PRIMARY,
            g,
            0,
        ));
    }
    if std::env::var("ATLAS_MTP_GATE_SERIAL_ARM").ok().as_deref() == Some("1") {
        arms.push(ArmSpec::serial("serial"));
    }

    if arms.len() < 2 {
        tracing::warn!(
            "--mtp-gate auto: only one arm is available ({primary_name}) — the gate will \
             measure but can never switch. Pass --dflash AND --speculative to build both \
             proposers, or set ATLAS_MTP_GATE_SERIAL_ARM=1 to arbitrate against plain decode."
        );
    }
    Some(MtpGate::new(arms))
}

/// Point the model at `arm`'s proposer and γ cap, moving every live sequence's
/// parked proposer state across with it.
///
/// The state swap is what makes an arm change safe: each proposer owns a
/// differently-typed per-sequence state and readers downcast
/// `seq.proposer_state` to the type they expect, so repointing the model
/// without moving the states would hand the incoming drafter the outgoing
/// one's buffers.
fn select_gate_arm(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    arm: ArmSpec,
) -> std::result::Result<(), String> {
    // `num_drafts` is read at the dispatch site, not here — this function only
    // repoints the model. The `..` keeps it that way without re-binding it.
    let ArmKind::Spec {
        proposer_arm,
        draft_cap,
        ..
    } = arm.kind
    else {
        // Serial arm: nothing to select. The γ override is left alone because
        // no propose will run this step.
        return Ok(());
    };
    if model.proposer_arm_kind(proposer_arm).is_none() {
        return Err(format!(
            "requested physical proposer arm {proposer_arm}, but it is not installed"
        ));
    }
    let previous = model.proposer_arm();
    if previous != proposer_arm {
        let effective = model.set_proposer_arm(proposer_arm);
        if effective != previous {
            for a in active.iter_mut() {
                model.swap_proposer_state(&mut a.seq);
            }
        }
        if effective != proposer_arm {
            return Err(format!(
                "requested physical proposer arm {proposer_arm}, model selected {effective}"
            ));
        }
    }
    spark_model::speculative::set_dflash_gamma_override(draft_cap);
    Ok(())
}

fn fail_closed_scheduler_transition(
    active: &mut [ActiveSeq],
    cache_on_finish: &mut [bool],
    transition: &str,
    error: &dyn std::fmt::Display,
) {
    tracing::error!("{transition}: {error}; finishing affected sequences fail-closed");
    for (idx, a) in active.iter_mut().enumerate() {
        proposal_lifecycle::clear(
            &mut a.pending_drafts,
            &mut a.pending_tree_payload,
            &mut a.pending_dflash_sampled_proposal,
        );
        a.finished = true;
        if let Some(cache) = cache_on_finish.get_mut(idx) {
            *cache = false;
        }
    }
}

pub(crate) fn speculation_routes(
    force_disable_speculation: bool,
    has_internal_spec: bool,
    use_speculative: bool,
    has_proposer: bool,
) -> (bool, bool) {
    if force_disable_speculation {
        (false, false)
    } else {
        (has_internal_spec, use_speculative && has_proposer)
    }
}

/// Startup receipt for a scheduler with no speculative route. Without it a
/// target-only launch (FLUX_DISABLE_SPECULATION=1, FLUX_SPEC_MODE=none or a
/// target-only launch profile: no proposer is built) logged nothing, because
/// the only `scheduler_path=ordinary-target-only` receipts belong to the
/// loaded-serial control and the TC-parity reference.
fn target_only_receipt(
    use_internal_spec: bool,
    num_drafts: usize,
    proposer_loaded: bool,
    force_disable_speculation: bool,
) -> Option<String> {
    if use_internal_spec || num_drafts > 0 {
        return None;
    }
    let control = if force_disable_speculation {
        "disabled"
    } else if proposer_loaded {
        "not-requested"
    } else {
        "unloaded"
    };
    Some(format!(
        "SPECULATION_CONTROL_RECEIPT schema=1 control={control} proposer_loaded={proposer_loaded} scheduler_path=ordinary-target-only use_mtp=false num_drafts=0"
    ))
}

/// Run the scheduler loop on the current thread.
#[allow(clippy::too_many_arguments)]
pub fn run(
    model: Box<dyn Model>,
    request_rx: tokio::sync::mpsc::Receiver<InferenceRequest>,
    eos_tokens: Vec<u32>,
    max_batch_size: usize,
    use_speculative: bool,
    num_drafts: usize,
    policy: Box<dyn SchedulingPolicy>,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    // Resolved `--prefill-chunk-with-active-decode` (see
    // `prefill_chunk_policy`): `None` keeps every chunk at `max_prefill_tokens`.
    prefill_chunk_with_active_decode: Option<usize>,
    // `--decode-gap-target-ms` (see `prefill_slo`): 0 disables the
    // SLO-budgeted prefill interleave.
    decode_gap_target_ms: u32,
    use_self_speculative: bool,
    use_ngram_speculative: bool,
    swap_space_gb: usize,
    high_speed_swap_cfg: Option<spark_storage::HighSpeedSwapConfig>,
    block_size: usize,
    max_seq_len: usize,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    reflection_suppress_ids: Vec<u32>,
    mut grammar_engine: Option<GrammarEngine>,
    adaptive_sampling: bool,
    mut session_manager: crate::session_manager::SessionSsmManager,
    spontaneous_think_budget: u32,
    // Resolved `--mtp-gate` value; `None` (the default) leaves the speculation
    // gate disarmed and the run byte-identical to before. (A plain comment,
    // not a doc comment: rustc rejects `///` on a function parameter.)
    mtp_gate_mode: Option<String>,
    // Strictly parsed once by serve_load before model construction and moved
    // into this thread.  This remains process-global across every Model
    // implementation, including DSpark models that use the trait default.
    force_disable_speculation: bool,
) {
    qwen4_batch_route::configure(max_batch_size);
    match spec_timing::configure(max_batch_size) {
        Ok(true) => tracing::info!("DFlash spec-cycle schema 2 timing armed (C=1, async=0)"),
        Ok(false) => {}
        Err(error) => tracing::error!("DFlash spec-cycle schema 2 timing disabled: {error}"),
    }
    model
        .bind_gpu_to_thread()
        .expect("Failed to bind CUDA context to scheduler thread");
    // Diagnostic/correctness escape hatch: keep the proposer and all of its
    // model-side buffers loaded, but route every request through the ordinary
    // decode scheduler.  `serve_load` parsed this value strictly before model
    // construction and moved it into this thread, so it cannot change
    // state-machine arms in the middle of a sequence.
    let has_internal_spec = model.has_internal_spec();
    let (use_internal_spec, use_mtp) = speculation_routes(
        force_disable_speculation,
        has_internal_spec,
        use_speculative,
        model.has_proposer(),
    );
    if force_disable_speculation && ((use_speculative && model.has_proposer()) || has_internal_spec)
    {
        tracing::warn!(
            "ATLAS_DISABLE_SPECULATION=1: proposer remains loaded; using ordinary decode scheduling"
        );
        if use_speculative
            && model.proposer_arm_count() == 1
            && model.proposer_arm_kind(0)
                == Some(spark_model::speculative::DraftProposerKind::Qwen4NativeMtp)
        {
            tracing::info!(
                "QWEN4_SPECULATION_CONTROL_RECEIPT schema=1 control=loaded-serial proposer_loaded=true proposer_kind=qwen4-native-mtp scheduler_path=ordinary-target-only use_mtp=false num_drafts=0"
            );
        }
    }
    if use_internal_spec {
        tracing::info!(
            "Internal DSpark speculative decoding: ENABLED (actual scheduler dispatch: Model::spec_verify/spec_commit; exact greedy requests only)"
        );
        tracing::warn!(
            "Internal DSpark sampled decoding: HOLD (temperature > 0 routes to ordinary decode until exact target-p/draft-q acceptance is qualified)"
        );
    }
    let num_drafts = if use_mtp || use_self_speculative || use_ngram_speculative {
        num_drafts.max(1)
    } else {
        0
    };
    if let Some(receipt) = target_only_receipt(
        use_internal_spec,
        num_drafts,
        model.has_proposer(),
        force_disable_speculation,
    ) {
        tracing::info!("{receipt}");
    }
    let chunked = max_prefill_tokens > 0;
    let mut ngram_proposer = if use_ngram_speculative {
        Some(NgramProposer::new(4)) // 4-gram context
    } else {
        None
    };
    // True exactly while the secondary stream owns an n-gram checkpoint or
    // rollback receipt. It is independent of `pending_drafts`: a rejected
    // frame with no next proposal still has a rollback that must be ordered
    // before serial decode or sequence teardown.
    let mut ngram_sync_pending = false;
    // Self-spec's accepted recurrent state is committed on the same secondary
    // stream.  Keep its receipt independent so no mode transition or request
    // retirement can reuse an unordered state image.
    let mut self_spec_sync_pending = false;
    // Throughput-arbitrated speculation gate. `None` unless `--mtp-gate` was
    // passed, so the default path below is exactly the pre-gate `step_mtp`
    // call. `last_gate_arm` tracks the arm the PREVIOUS step actually ran, so
    // the transition cleanup fires on probe excursions too and not only on
    // committed switches.
    let primary_proposer_is_dflash = model.proposer_is_dflash();
    let mut spec_gate = build_speculation_gate(&*model, use_mtp, mtp_gate_mode.as_deref());
    let entry_pin =
        mtp_gate::parse_entry_pin_tokens(std::env::var("ATLAS_SPEC_ENTRY_PIN").ok().as_deref());
    if use_mtp {
        tracing::info!(
            env = "ATLAS_SPEC_ENTRY_PIN",
            tokens = entry_pin.tokens,
            source = entry_pin.source,
            gate_armed = spec_gate.is_some(),
            "spec-entry dispatch provenance"
        );
    }
    // Hoisted out of the step loop: the arm set is fixed at construction, so
    // "can this gate ever change what we run?" is a constant for the whole
    // serve. False for `--mtp-gate dflash|mtp` (a pinned arm) and for `auto`
    // on a build that only produced one proposer. When false the scheduler
    // skips ALL of the gate's measurement below — see the step dispatch.
    let gate_arbitrates = spec_gate.as_ref().is_some_and(MtpGate::arbitrates);
    let mut last_gate_arm: Option<usize> = None;
    // A Serial arm leaves the previously selected proposer installed. Keep its
    // own request width as well: the DFlash primary normally uses γ=15 while
    // the native MTP alternate uses one draft.
    let mut last_spec_num_drafts = num_drafts;
    spark_runtime::sampler::sampler_tie_order_receipt();
    let mut last_step_was_entry_pin = false;
    tracing::info!(
        "Scheduler started (batched mode, max_batch={max_batch_size}, mtp={}, ngram={}, num_drafts={num_drafts}, policy={}, chunked_prefill={}, max_prefill_tokens={}, prefill_chunk_with_active_decode={})",
        use_mtp,
        use_ngram_speculative,
        policy.name(),
        chunked,
        if chunked { max_prefill_tokens } else { 0 },
        prefill_chunk_with_active_decode.unwrap_or(0),
    );
    if let Some(cap) = prefill_chunk_with_active_decode {
        tracing::info!(
            "PREFILL_CHUNK_WITH_ACTIVE_DECODE cap={cap} full_chunk={max_prefill_tokens}: chunks \
             issued while another sequence decodes are capped; prefill numerics for those \
             requests differ from an uncapped run (chunk 2+ reads earlier context from the KV cache)"
        );
    }
    // ATLAS_PREFILL_INTERLEAVE=1 (default off): explicit legacy step-ratio override.
    let mut legacy_interleave = prefill_interleave::PrefillInterleave::from_env(
        max_prefill_tokens,
        model.hc_mult(),
        model.is_mla(),
    );
    let prepared_widths = model.fast_prefill_chunk_widths();
    let parity_width = prefill_slo::parse_parity_width(
        std::env::var(prefill_slo::PARITY_WIDTH_ENV).ok().as_deref(),
        &prepared_widths,
    )
    .unwrap_or_else(|error| {
        tracing::error!("{error}; parity diagnostic disabled");
        None
    });
    let effective_decode_gap_target_ms = if legacy_interleave.is_enabled() {
        0
    } else {
        decode_gap_target_ms
    };
    let mut slo_interleave = prefill_slo::SloRuntime::new(
        &prepared_widths,
        effective_decode_gap_target_ms,
        chunked,
        model.is_mla(),
        block_size,
    )
    .with_parity_width(parity_width);
    if !legacy_interleave.is_enabled() {
        tracing::info!("{}", slo_interleave.startup_message());
    }
    if let Some(width) = slo_interleave.parity_width() {
        tracing::warn!(
            "{}={width}: DIAGNOSTIC: every prompt is prefilled in bounded {width}-token chunks \
             even with no decode row (parity testing only; numerics differ from full chunks)",
            prefill_slo::PARITY_WIDTH_ENV
        );
    }
    if let Some(warning) = prefill_chunk_policy::fast_path_warning(
        prefill_chunk_with_active_decode,
        model.fast_prefill_chunk_rows(),
        slo_interleave.engaged(),
    ) {
        tracing::warn!("{warning}");
    }

    // Reserve the request's remaining decode budget, not only its prompt,
    // before admitting it into the fixed-size paged-KV pool. Resolve the
    // operator watermark once so admission policy cannot drift mid-run.
    let admit_watermark = admission::resolve_admit_watermark(max_seq_len);
    // Serial and batched native K5 both reserve each request's draft rows.
    let verify_lookahead = admission::resolve_verify_lookahead_tokens(
        spark_model::model::qwen4_k5_cmulti::k5_cmulti_selected() == Ok(true)
            || spark_model::layers::qwen4_verify_lanes::batched_k5_selected().unwrap_or(false),
        use_mtp
            && model.proposer_arm_kind(0)
                == Some(spark_model::speculative::DraftProposerKind::Qwen4NativeMtp),
        num_drafts,
    );

    // `--kv-pool-tokens`: reserve each row's next KV positions before the
    // batched step so pool exhaustion retires one request, not the batch.
    // `None` for the per-slot pool, which leaves the tick unchanged.
    let kv_row_headroom = kv_headroom::row_headroom(
        model.num_total_blocks(),
        max_batch_size,
        max_seq_len,
        block_size,
        num_drafts,
    );
    if let Some(rows) = kv_row_headroom {
        tracing::info!(
            "KV_POOL shared: {} blocks for {max_batch_size} slot(s) at max_seq_len {max_seq_len}; \
             each tick reserves {rows} KV position(s) per row first, so exhaustion retires one \
             request, not the batch",
            model.num_total_blocks()
        );
    }

    let pending = Arc::new((
        Mutex::new(PendingQueue {
            requests: Vec::new(),
            closed: false,
        }),
        Condvar::new(),
    ));

    // Receiver thread: moves requests from the bounded tokio channel into
    // the pending queue and parks at `pending_queue_cap`, so the channel
    // bound finally backpressures the HTTP handlers (M1).
    // `--max-queue-depth` is the waiting bound when set; the pending queue
    // must be able to hold that many so the gate, not this cap, answers 429.
    let _receiver = spawn_request_receiver(
        request_rx,
        Arc::clone(&pending),
        pending_queue_cap(max_batch_size).max(queue_gate::QUEUE_GATE.limit().unwrap_or(0)),
    );

    // Dedicated CUDA stream + event for prefill compute-copy overlap.
    let prefill_stream = model
        .create_stream()
        .expect("Failed to create prefill CUDA stream");
    let prefill_event = model
        .create_event()
        .expect("Failed to create prefill CUDA event");

    let mut active: Vec<ActiveSeq> = Vec::new();
    let mut prefilling: Vec<PrefillInProgress> = Vec::new();
    let mut swapped: Vec<SwappedSeq> = Vec::new();
    let mut spill_manager: Option<KvSpillManager> = if swap_space_gb > 0 {
        let max_bytes = swap_space_gb as u64 * 1024 * 1024 * 1024;
        match KvSpillManager::new(PathBuf::from("/tmp/flux-swap"), max_bytes) {
            Ok(mgr) => {
                tracing::info!("Swap space: {swap_space_gb} GB at /tmp/flux-swap/");
                Some(mgr)
            }
            Err(e) => {
                tracing::error!("Failed to initialize swap space: {e:#}");
                None
            }
        }
    } else {
        None
    };

    install_high_speed_swap(&*model, high_speed_swap_cfg);

    // Native MTP verifies DURING `<think>` spans by default. DFlash keeps the
    // ATLAS_THINK_SPEC=1 opt-in because its wide forward is a distinct
    // numerical path. The post-verify accept filter
    // (`think_spec_accept::dflash_thinking_accept`) re-derives the
    // plain-path token per position, so output stays byte-identical to
    // `step_decode_only`. Disqualified when the model emits fp32 decode
    // logits (the filter's row D2H assumes BF16). Adaptive sampling is
    // threaded through the same per-row function, including state updates.
    // FP32 verify logits remain serial because the current per-row oracle reads
    // BF16. `dflash_spec_think` is resolved once so request scheduling cannot
    // change if the environment is mutated mid-serve.
    let dflash_spec_think = think_spec_enabled();
    let think_ctx = ThinkSpecCtx {
        enabled: !model.decode_logits_fp32(),
        code_fence_token,
        reflection_suppress_ids: &reflection_suppress_ids,
        adaptive_sampling,
        // Model swaps refresh this atomic after constructing a new target.
        // Read it at each policy decision rather than snapshotting scheduler
        // startup state (which can race the loader or inherit the old model).
        honor_eos_inside_thinking: None,
    };
    // GLM-5.3 speculation runs its own policy-before-commit verifier; every
    // other model takes `step_mtp` unchanged.
    let glm_policy = model.requires_verify_policy();
    let glm_settings = glm53_policy_driver::DecodeSettings {
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        reflection_suppress_ids: &reflection_suppress_ids,
        adaptive_sampling,
    };
    let mut snapshot_steps: u64 = 0;
    let mut last_session_eviction = Instant::now();
    const SESSION_EVICTION_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);
    let mut ple_queue_prefetch = ple_queue_prefetch::QueuePrefetch::from_env();
    loop {
        let legacy_tick = legacy_interleave.is_enabled().then(|| {
            legacy_interleave.plan_tick(
                Instant::now(),
                active.len(),
                prefilling.len(),
                max_prefill_tokens,
            )
        });
        let prefill_chunks_before = prefill_interleave::prefill_chunks_recorded();
        let tick_chunk_tokens = legacy_tick.as_ref().map_or(
            max_prefill_tokens,
            prefill_interleave::TickPlan::chunk_tokens,
        );
        // SLO interleave: decide this tick's chunk for the prompt mid-prefill
        // (else the head of the queue). While a row decodes, a new prompt is
        // admitted only on a chunk tick with nothing mid-prefill.
        let slo_candidate = prefilling
            .first()
            .map(|p| prefill_slo::Candidate {
                offset: p.chunk_offset,
                remaining: p.prompt_tokens.len().saturating_sub(p.chunk_offset),
            })
            .or_else(|| {
                pending
                    .0
                    .lock()
                    .requests
                    .first()
                    .map(|r| prefill_slo::Candidate {
                        offset: 0,
                        remaining: r.prompt_len(),
                    })
            });
        let tick_plan = slo_interleave.plan_tick(
            slo_candidate,
            !active.is_empty(),
            prefill_chunk_with_active_decode,
            tick_chunk_tokens,
            model.is_mla(),
        );
        let admit_batch_cap = match legacy_tick.as_ref() {
            Some(tick) => tick.drain_limit(max_batch_size, active.len(), prefilling.len()),
            None => tick_plan.admission_batch_cap(max_batch_size, active.len(), prefilling.len()),
        };
        // ── Drain pending → start prefill (chunked or full) ──
        let new_reqs = drain_pending_requests_for_tick(
            &pending,
            &active,
            &prefilling,
            &*policy,
            admit_batch_cap,
            tick_plan.plan.widths.is_some(),
        );
        let new_reqs = admission::gate_admissions(
            &*model,
            &pending,
            new_reqs,
            &active,
            &prefilling,
            &swapped,
            admit_watermark,
            max_seq_len,
            block_size,
            verify_lookahead,
        );
        // Publish only when the dashboard is active. Plain/benchmark mode
        // pays one atomic branch and never takes the snapshot mutex.
        if crate::tui::is_active() {
            snapshot_steps += 1;
            let (mtp_mode, delivered_tps) = match spec_gate.as_ref() {
                Some(g) => g.observe(),
                None => (snapshot::MtpModeSnap::Off, 0.0),
            };
            snapshot::publish(snapshot::SchedulerSnapshot {
                active_seqs: active.len() as u32,
                prefilling_seqs: prefilling.len() as u32,
                swapped_seqs: swapped.len() as u32,
                pending_len: pending.0.lock().requests.len() as u32,
                kv_blocks_free: model.num_free_blocks() as u32,
                kv_blocks_total: model.num_total_blocks() as u32,
                ssm_slots_used: session_manager.session_count() as u32,
                ssm_slots_total: session_manager.total_slots() as u32,
                mtp_mode,
                delivered_tps,
                steps_total: snapshot_steps,
                published_at: std::time::Instant::now(),
            });
        }
        if new_reqs.is_empty() && active.is_empty() && prefilling.is_empty() {
            // Idle: report acceptance frames not yet summarized (no-op
            // unless ATLAS_MTP_ACCEPT_LOG=1 and new frames exist).
            spec_accept_log::flush_committed_summaries();
            // Receiver thread was closed (shutdown).
            let pending_closed = pending.0.lock().closed;
            if pending_closed {
                break;
            }
        }

        // ── Swap-out: evict active sequences to disk when blocks run low ──
        if let Some(ref mut spill) = spill_manager {
            for req in &new_reqs {
                let prompt_len = req.prompt_len();
                let blocks_needed = prompt_len / block_size + 1;
                while model.num_free_blocks() < blocks_needed && !active.is_empty() {
                    let victim_idx = active
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| a.grammar_state.is_none())
                        .max_by_key(|(_, a)| a.seq.block_table.len())
                        .map(|(i, _)| i);
                    let Some(victim_idx) = victim_idx else {
                        tracing::warn!("No swappable sequences (all grammar-active)");
                        break;
                    };
                    match swap_out_sequence(&*model, &mut active, victim_idx, spill) {
                        Ok(s) => {
                            tracing::info!(
                                "Swap-out: evicted seq (seq_len={}, blocks={}) to disk",
                                s.seq_len,
                                s.num_blocks,
                            );
                            swapped.push(s);
                        }
                        Err(e) => {
                            tracing::error!("Swap-out failed: {e:#}");
                            break;
                        }
                    }
                }
            }
        }

        // ── Start new requests ──
        let tick_start = std::time::Instant::now();
        let chunk_plan = tick_plan.plan;
        let mut tick_ledger = prefill_slo::PrefillTickLedger::default();
        let started = new_reqs.len();
        start_new_requests(
            &*model,
            new_reqs,
            &pending,
            &swapped,
            admit_watermark,
            max_seq_len,
            block_size,
            verify_lookahead,
            max_batch_size,
            chunked,
            tick_chunk_tokens,
            max_batch_tokens,
            chunk_plan,
            &eos_tokens,
            prefill_stream,
            prefill_event,
            &mut grammar_engine,
            spontaneous_think_budget,
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            &mut active,
            &mut prefilling,
            &mut tick_ledger,
        );

        // ── Continue in-progress prefills ──
        // SLO interleave: at most one chunk between decode steps (not on a
        // skip tick, not after chunk 0 of a new prompt ran this tick).
        let continue_prefill = legacy_tick
            .as_ref()
            .is_none_or(prefill_interleave::TickPlan::continue_allowed)
            && tick_plan.continue_allowed(&tick_ledger);
        let did_mixed_step = if continue_prefill {
            continue_in_progress_prefills(
                &*model,
                &*policy,
                &mut active,
                &mut prefilling,
                tick_chunk_tokens,
                chunk_plan,
                prefill_stream,
                prefill_event,
                use_mtp,
                use_self_speculative,
                use_ngram_speculative,
                think_end_token,
                think_start_token,
                code_fence_token,
                tool_call_start_token,
                tool_call_end_token,
                &reflection_suppress_ids,
                adaptive_sampling,
                &mut tick_ledger,
            )
        } else {
            // Not this tick's prefill slot; cancelled or expired prefills
            // still retire now.
            retire_stopped_prefills(&*model, &mut prefilling, prefill_stream);
            false
        };
        // ATLAS_QWEN4_PLE_QUEUE_PREFETCH: while a request is running, start the
        // next one's chunk-0 PLE reads (same budget start_new_requests uses).
        if ple_queue_prefetch.selected() {
            let budget = tick_chunk_tokens.min(max_batch_tokens);
            let pending_guard = pending.0.lock();
            ple_queue_prefetch.tick(
                &*model,
                !active.is_empty() || !prefilling.is_empty(),
                pending_guard
                    .requests
                    .first()
                    .map(|req| req.prompt_tokens()),
                budget,
            );
        }
        if let Some(ref tick) = legacy_tick {
            legacy_interleave.finish_tick(
                tick,
                prefill_interleave::chunks_run(
                    prefill_chunks_before,
                    prefill_interleave::prefill_chunks_recorded(),
                    started,
                    chunked,
                ),
                !active.is_empty(),
            );
        }

        if active.is_empty() {
            continue;
        }
        if let Some(rows) = kv_row_headroom {
            kv_headroom::finish_rows_without_kv_headroom(&*model, &mut active, rows);
            if active.is_empty() {
                continue;
            }
        }

        // Retirement consumes this in lockstep with `active`. Ordinary,
        // bootstrap and serial finishes stay cacheable; `step_mtp` clears only
        // entries that finish inside an over-planned speculative verify frame.
        let mut cache_on_finish = vec![true; active.len()];

        // Skip decode when mixed_forward already processed decode logits.
        if !did_mixed_step {
            // Ensure any in-flight prefill work on the prefill stream is complete
            // before decode starts on the default stream.
            if !prefilling.is_empty() {
                prefill_stream_barrier::complete_prefill_handoff(
                    &*model,
                    prefill_stream,
                    prefill_event,
                    "scheduler prefill-to-decode handoff",
                );
            }

            if use_internal_spec {
                if active.len() == 1
                    && internal_spec_eligible(&active[0], &*model, adaptive_sampling)
                {
                    // DeepSeek-V4.1 DSpark: model-owned draft/verify/rollback.
                    step_internal_spec(
                        &*model,
                        &mut active,
                        &mut cache_on_finish,
                        think_end_token,
                        think_start_token,
                        code_fence_token,
                        tool_call_start_token,
                        tool_call_end_token,
                        &reflection_suppress_ids,
                        adaptive_sampling,
                    );
                } else {
                    // Internal-spec models own their entire speculation mode.
                    // Ineligible/sampled/batched requests HOLD on serial decode
                    // and may never fall through to an unrelated external path.
                    step_decode_only(
                        &*model,
                        &mut active,
                        &mut cache_on_finish,
                        think_end_token,
                        think_start_token,
                        code_fence_token,
                        tool_call_start_token,
                        tool_call_end_token,
                        &reflection_suppress_ids,
                        adaptive_sampling,
                    );
                }
            } else if use_ngram_speculative {
                let eligible = active.len() == 1
                    && active[0].grammar_state.is_none()
                    && (!active[0].inside_thinking || think_ctx.enabled)
                    && !think_nudge::injection_pending(&active[0]);
                if eligible {
                    // N-gram speculative: CPU proposer + CUDA-graphed K=2 verify.
                    if let Some(ref mut proposer) = ngram_proposer {
                        let was_verify = !active[0].pending_drafts.is_empty();
                        step_ngram(
                            &*model,
                            &mut active,
                            proposer,
                            &mut ngram_sync_pending,
                            &think_ctx,
                        );
                        if was_verify && active[0].finished {
                            cache_on_finish[0] = false;
                        }
                    }
                } else {
                    // Grammar, batching, or an unavailable thinking-policy
                    // oracle makes this frame ineligible. Drop its proposal and
                    // drain the paired async receipt before serial decode.
                    let synced = abandon_ngram_frame(&*model, &mut active, &mut ngram_sync_pending);
                    if synced {
                        step_decode_only(
                            &*model,
                            &mut active,
                            &mut cache_on_finish,
                            think_end_token,
                            think_start_token,
                            code_fence_token,
                            tool_call_start_token,
                            tool_call_end_token,
                            &reflection_suppress_ids,
                            adaptive_sampling,
                        );
                    }
                }
                if active.iter().any(|a| a.finished) {
                    abandon_ngram_frame(&*model, &mut active, &mut ngram_sync_pending);
                }
            } else if use_self_speculative {
                let eligible = active.len() == 1
                    && active[0].grammar_state.is_none()
                    && (!active[0].inside_thinking || think_ctx.enabled)
                    && !think_nudge::injection_pending(&active[0]);
                if eligible {
                    // Self-speculative: draft via layer-skipping, verify with
                    // the full model, then commit recurrent state asynchronously.
                    step_self_spec(
                        &*model,
                        &mut active,
                        num_drafts,
                        &mut self_spec_sync_pending,
                        &think_ctx,
                    );
                } else if abandon_self_spec_frame(&*model, &mut active, &mut self_spec_sync_pending)
                {
                    // Grammar, batching, or an unavailable thinking-policy
                    // oracle is a serial-only frame.  The async commit receipt
                    // above must be ordered before that fallback may run.
                    step_decode_only(
                        &*model,
                        &mut active,
                        &mut cache_on_finish,
                        think_end_token,
                        think_start_token,
                        code_fence_token,
                        tool_call_start_token,
                        tool_call_end_token,
                        &reflection_suppress_ids,
                        adaptive_sampling,
                    );
                }
                if active.iter().any(|a| a.finished) {
                    abandon_self_spec_frame(&*model, &mut active, &mut self_spec_sync_pending);
                    // The self-spec helper may have over-planned a dense verify
                    // frame before the terminal emission. Conservatively skip
                    // final cache insertion for every finished request.
                    for (a, cacheable) in active.iter().zip(cache_on_finish.iter_mut()) {
                        if a.finished {
                            *cacheable = false;
                        }
                    }
                }
            } else if use_mtp {
                let thinking_step_allowed = if let Some(gate) = spec_gate.as_ref() {
                    mtp_gate::arm_allows_thinking(
                        gate.next_arm().kind,
                        primary_proposer_is_dflash,
                        dflash_spec_think,
                        think_ctx.enabled,
                    )
                } else {
                    let current = ArmKind::Spec {
                        proposer_arm: if primary_proposer_is_dflash {
                            mtp_gate::PROPOSER_ARM_PRIMARY
                        } else {
                            model.proposer_arm()
                        },
                        draft_cap: 0,
                        num_drafts,
                    };
                    mtp_gate::arm_allows_thinking(
                        current,
                        primary_proposer_is_dflash,
                        dflash_spec_think,
                        think_ctx.enabled,
                    )
                };
                if active.iter().all(|a| {
                    // Native MTP may verify while thinking; DFlash requires
                    // ATLAS_THINK_SPEC=1. Both use the same post-verify policy
                    // oracle (see think_spec_accept.rs). suppress_tool_call /
                    // disable_mtp gates are unchanged.
                    (!a.inside_thinking || thinking_step_allowed)
                        && !a.suppress_tool_call
                        && !a.disable_mtp
                        // A thinking nudge is injected on plain decode only.
                        && !think_nudge::injection_pending(a)
                }) {
                    // MTP speculative decode for ALL active sequences.
                    //
                    // Concurrent-decode fix (2026-05-22): previously gated on
                    // `active.len() == 1`, which forced n>=2 into
                    // `step_decode_only`. That path runs the SSM
                    // `decode_multi_seq_inner` which sequentially calls
                    // per-seq `decode()` (see qwen3_ssm/trait_decode_multi_seq.rs
                    // — comment: "delegate to per-sequence single decode") with
                    // no CUDA graphs and no MTP, collapsing throughput to ~14
                    // tok/s aggregate at c=2 (down from 28 at c=1).
                    //
                    // `step_mtp` already loops bootstrap+verify across all
                    // active sequences (see mtp_step.rs lines 21,96). Each
                    // sequence reuses its own per-slot CUDA graph for K=3
                    // verify (graph cache is keyed on `seq.slot_idx` in
                    // verify_c.rs), so n>=2 captures one graph per slot on
                    // first iteration and replays on every subsequent step.
                    //
                    // Per-seq guards (inside_thinking / suppress_tool_call /
                    // disable_mtp) are checked across ALL active sequences
                    // because step_mtp doesn't conditionally bootstrap per
                    // active flag; if any seq disables MTP we fall back to
                    // batched decode_only for the whole batch this tick.
                    //
                    // ── Speculation gate (`--mtp-gate`) ──
                    //
                    // When armed, the gate picks which arm to run for THIS step —
                    // the external DFlash/DDTree drafter, the native MTP head, a
                    // γ-capped drafter, or plain decode — by comparing DELIVERED
                    // tok/s per arm over 16-step windows with hysteresis and
                    // periodic probing. Every arm emits real, correct tokens, so
                    // arbitration never wastes work. See `mtp_gate` for why this
                    // measures throughput and not draft acceptance.
                    //
                    // Disarmed (the default) this is exactly the previous call.
                    if let Some(gate) = spec_gate.as_mut() {
                        let arm_idx = gate.next_arm_index();
                        let arm = gate.next_arm();
                        let pin_active = matches!(arm.kind, ArmKind::Serial)
                            && mtp_gate::entry_pin_active(
                                entry_pin.tokens,
                                active.iter().map(|a| {
                                    (a.think_ended, a.inside_thinking, a.post_think_gate_steps)
                                }),
                            );
                        let pinned_spec_width = mtp_gate::entry_pin_spec_width(
                            arm.kind,
                            pin_active,
                            last_spec_num_drafts,
                        );
                        // Everything the gate measures exists to feed arbitration,
                        // so a gate that cannot arbitrate must not pay for it. A
                        // PINNED arm therefore skips the depth scan, both
                        // `seq_len` scans, the `Instant` pair and the window
                        // accounting, and what remains between it and the disarmed
                        // path is one `Option` compare plus a `match` on a value
                        // that never changes.
                        //
                        // `Some(before)` doubles as the "measure this step" flag.
                        let measure = if gate_arbitrates && pinned_spec_width.is_none() {
                            // One pass, two answers. Depth is the MAX `seq_len`
                            // (arm economics are depth-dependent: weight-bound at
                            // short context, KV/SSM-bound at depth). Delivered
                            // tokens are the SUM of committed outputs. `seq_len`
                            // is a target-input cursor and may include verify
                            // rows later rejected or rolled back. Counting only
                            // `active[0]` would
                            // under-report a speculative arm by a factor of n and
                            // bias the gate toward whichever arm is cheapest per
                            // step rather than fastest per token.
                            let width = active.len().max(1);
                            let (depth, before) =
                                active.iter().fold((0usize, 0usize), |(d, s), a| {
                                    (d.max(a.seq.seq_len), s + a.output_tokens.len())
                                });
                            gate.observe_depth(depth);
                            Some((before, width))
                        } else {
                            None
                        };

                        let mut transition_ok = true;
                        let arm_changed = last_gate_arm != Some(arm_idx);
                        if arm_changed {
                            // Arm transition. Drafts in flight belong to the
                            // OUTGOING arm and would be verified against the
                            // incoming one's assumptions, so drop them (and the
                            // paired DDTree payload) and order the previous
                            // verify's async live-state restore before the next
                            // step reads h_state/conv_state. The next speculative
                            // step then re-enters through `step_mtp`'s bootstrap
                            // phase, whose `trim_proposer_state(seq, 0, 0)` resets
                            // the drafter's RoPE conditioning — which is what
                            // makes re-entry lossless.
                            for a in active.iter_mut() {
                                proposal_lifecycle::clear(
                                    &mut a.pending_drafts,
                                    &mut a.pending_tree_payload,
                                    &mut a.pending_dflash_sampled_proposal,
                                );
                            }
                            if let Err(e) = model.sync_secondary() {
                                fail_closed_scheduler_transition(
                                    &mut active,
                                    &mut cache_on_finish,
                                    "gate arm switch sync_secondary",
                                    &e,
                                );
                                transition_ok = false;
                            } else if let Err(e) = select_gate_arm(&*model, &mut active, arm) {
                                fail_closed_scheduler_transition(
                                    &mut active,
                                    &mut cache_on_finish,
                                    "gate arm selection",
                                    &e,
                                );
                                transition_ok = false;
                            }
                            if transition_ok {
                                tracing::debug!(
                                    "speculation gate: running arm {} ({})",
                                    arm_idx,
                                    gate.arm_name(arm_idx)
                                );
                                last_gate_arm = Some(arm_idx);
                            }
                        } else if mtp_gate::entry_pin_exits_to_serial(
                            arm.kind,
                            last_step_was_entry_pin,
                            pinned_spec_width,
                        ) {
                            // The gate stayed on Serial while the entry pin ran the
                            // installed proposer, so no arm transition exists to
                            // drop that last pinned step's drafts or order its
                            // async recurrent-state restore.
                            for a in active.iter_mut() {
                                proposal_lifecycle::clear(
                                    &mut a.pending_drafts,
                                    &mut a.pending_tree_payload,
                                    &mut a.pending_dflash_sampled_proposal,
                                );
                            }
                            if let Err(e) = model.sync_secondary() {
                                fail_closed_scheduler_transition(
                                    &mut active,
                                    &mut cache_on_finish,
                                    "entry-pin→serial sync_secondary",
                                    &e,
                                );
                                transition_ok = false;
                            }
                        }

                        // Taken as late as possible so the one-shot transition
                        // work above is not charged to the arm being switched TO.
                        let t0 = if transition_ok {
                            measure.map(|_| Instant::now())
                        } else {
                            None
                        };
                        if !transition_ok {
                            // The sequences were finished above. Do not execute
                            // or credit the arm whose transition failed.
                        } else if let Some(nd) = pinned_spec_width {
                            // Entry pins are correctness dispatches, not samples of
                            // the Serial arm. Run the proposer which remains selected
                            // across the serial dwell and leave gate statistics
                            // untouched. `step_mtp`'s post-think policy gate routes
                            // both native short drafts and DFlash γ blocks through
                            // the row-by-row sampler oracle.
                            if glm_policy {
                                glm53_policy_driver::step(
                                    &*model,
                                    &mut active,
                                    &mut cache_on_finish,
                                    nd,
                                    &glm_settings,
                                );
                            } else {
                                step_mtp(
                                    &*model,
                                    &mut active,
                                    nd,
                                    &think_ctx,
                                    &mut cache_on_finish,
                                );
                            }
                        } else {
                            match arm.kind {
                                ArmKind::Serial => {
                                    step_decode_only(
                                        &*model,
                                        &mut active,
                                        &mut cache_on_finish,
                                        think_end_token,
                                        think_start_token,
                                        code_fence_token,
                                        tool_call_start_token,
                                        tool_call_end_token,
                                        &reflection_suppress_ids,
                                        adaptive_sampling,
                                    );
                                }
                                ArmKind::Spec {
                                    num_drafts: arm_drafts,
                                    ..
                                } => {
                                    // 0 means "the run's configured value" — the
                                    // DFlash arm — so only the alternate arm departs
                                    // from it.
                                    let nd = if arm_drafts == 0 {
                                        num_drafts
                                    } else {
                                        arm_drafts
                                    };
                                    last_spec_num_drafts = nd;
                                    if glm_policy {
                                        glm53_policy_driver::step(
                                            &*model,
                                            &mut active,
                                            &mut cache_on_finish,
                                            nd,
                                            &glm_settings,
                                        );
                                    } else {
                                        step_mtp(
                                            &*model,
                                            &mut active,
                                            nd,
                                            &think_ctx,
                                            &mut cache_on_finish,
                                        );
                                    }
                                }
                            }
                        }
                        if let (Some((before, width)), Some(t0)) = (measure, t0) {
                            let after: usize = active.iter().map(|a| a.output_tokens.len()).sum();
                            gate.record_step(t0.elapsed(), after.saturating_sub(before), width);
                            // Drained rather than acted on: the arm change is
                            // applied from `next_arm_index` at the top of the next
                            // step (which also covers probe excursions, where no
                            // "switch" is reported). Taking it here keeps the
                            // one-shot from going stale and gives the switch a
                            // single owner. Only reachable when arbitrating —
                            // a pinned gate never produces a switch to drain.
                            let _ = gate.take_fresh_switch();
                        }
                        last_step_was_entry_pin = pinned_spec_width.is_some();
                    } else if glm_policy {
                        glm53_policy_driver::step(
                            &*model,
                            &mut active,
                            &mut cache_on_finish,
                            num_drafts,
                            &glm_settings,
                        );
                    } else {
                        step_mtp(
                            &*model,
                            &mut active,
                            num_drafts,
                            &think_ctx,
                            &mut cache_on_finish,
                        );
                    }
                } else {
                    // Fall through to the ordinary decode path below (`use_mtp` is true here).
                    // Clear stale drafts and order the async verify-commit secondary-stream
                    // restore before plain decode reads h_state/conv_state.
                    let mut transition_ok = true;
                    for a in active.iter_mut() {
                        proposal_lifecycle::clear(
                            &mut a.pending_drafts,
                            &mut a.pending_tree_payload,
                            &mut a.pending_dflash_sampled_proposal,
                        );
                    }
                    if let Err(e) = model.sync_secondary() {
                        fail_closed_scheduler_transition(
                            &mut active,
                            &mut cache_on_finish,
                            "mtp→decode sync_secondary",
                            &e,
                        );
                        transition_ok = false;
                    }
                    last_step_was_entry_pin = false;
                    if transition_ok {
                        step_decode_only(
                            &*model,
                            &mut active,
                            &mut cache_on_finish,
                            think_end_token,
                            think_start_token,
                            code_fence_token,
                            tool_call_start_token,
                            tool_call_end_token,
                            &reflection_suppress_ids,
                            adaptive_sampling,
                        );
                    }
                }
            } else {
                // Batch decode (no MTP).
                step_decode_only(
                    &*model,
                    &mut active,
                    &mut cache_on_finish,
                    think_end_token,
                    think_start_token,
                    code_fence_token,
                    tool_call_start_token,
                    tool_call_end_token,
                    &reflection_suppress_ids,
                    adaptive_sampling,
                );
            }
        }

        // The decode step's logits D2H waited for this tick's prefill chunk.
        slo_interleave.observe_tick(
            tick_ledger,
            !did_mixed_step,
            tick_start.elapsed().as_secs_f64() * 1000.0,
        );
        retire_finished_sequences(&*model, &mut active, &mut cache_on_finish);

        // ── Swap-in: resume swapped sequences when blocks free up ──
        if let Some(ref mut spill) = spill_manager {
            let mut resumed_any = true;
            while resumed_any && !swapped.is_empty() && active.len() < max_batch_size {
                resumed_any = false;
                let free = model.num_free_blocks();
                if let Some(idx) = swapped.iter().position(|s| s.num_blocks <= free) {
                    let s = swapped.remove(idx);
                    match resume_swapped_seq(think_end_token, think_start_token, &*model, s, spill)
                    {
                        Ok(a) => {
                            tracing::info!(
                                "Swap-in: restored seq (seq_len={}, blocks={})",
                                a.seq.seq_len,
                                a.seq.block_table.len(),
                            );
                            active.push(a);
                            resumed_any = true;
                        }
                        Err(e) => {
                            tracing::error!("Swap-in failed: {e:#}");
                        }
                    }
                }
            }
        }

        // Periodic session eviction: free SSM snapshots for expired sessions.
        if session_manager.session_count() > 0
            && last_session_eviction.elapsed() >= SESSION_EVICTION_INTERVAL
        {
            last_session_eviction = Instant::now();
            let freed_slots = session_manager.evict_expired();
            if !freed_slots.is_empty() {
                tracing::info!(
                    "Session eviction: freed {} SSM snapshot slot(s), {} sessions active",
                    freed_slots.len(),
                    session_manager.session_count()
                );
            }
        }
    }

    // Drain any remaining active sequences on shutdown.
    for mut a in active {
        finish_sequence(&*model, &mut a);
    }
    if let Some(ref mut spill) = spill_manager {
        for s in swapped {
            let _ = spill.remove_file(s.swap_id);
        }
    }
    for p in prefilling {
        let mut seq = p.seq;
        let _ = model.free_sequence(&mut seq);
        let _ = model.ep_broadcast_cmd(0xFFFFFFF1);
    }
    let _ = model.ep_broadcast_cmd(0xFFFFFFFF);
    spec_accept_log::flush_committed_summaries();
    tracing::info!("Scheduler stopped");
}
