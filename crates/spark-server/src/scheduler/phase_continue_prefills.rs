// SPDX-License-Identifier: AGPL-3.0-only

//! Phase: continue in-progress chunked prefills. When `active` is empty,
//! all chunks run back-to-back (TTFT minimisation). When active is
//! nonempty, exactly one chunk runs per scheduler iteration to bound
//! TPOT — except when mixed_forward fuses a prefill chunk + decode in a
//! single pass.
//!
//! Returns `did_mixed_step` so the caller can skip the standalone decode
//! call (mixed forward already processed decode logits).
//!
//! Layout: this file is the dispatcher only; the three per-path bodies
//! live in the sibling sub-modules under `phase_continue_prefills/`:
//!
//!  - `run_standard`        — single-stream chunked-prefill body
//!                            (mixed_forward or plain prefill_chunk).
//!  - `run_batched_prefill` — N-stream batched-prefill step.
//!  - `run_batched_mixed`   — batched mixed (decode+prefill) step.

#[path = "phase_continue_prefills/run_batched_mixed.rs"]
mod run_batched_mixed;
#[path = "phase_continue_prefills/run_batched_prefill.rs"]
mod run_batched_prefill;
#[path = "phase_continue_prefills/run_standard.rs"]
mod run_standard;

use std::time::Instant;

use spark_model::traits::Model;

use super::phase_promote_prefills::promote_completed_prefills;
use super::prefill_a_step::prefill_first_token_applies_grammar;
use super::prefill_lifecycle::{retire_stopped_prefills, stop_allocated_prefill};
use super::prefill_stream_barrier::{complete_prefill_handoff, drain_before_prefill_cleanup};
use super::sample_token_with_logprobs_and_grammar;
use super::types::{ActiveSeq, PrefillInProgress};
use crate::scheduling_policy::{ActiveSeqTiming, SchedulingPolicy};

use run_batched_mixed::run_batched_mixed_step;
use run_batched_prefill::run_batched_prefill_step;
use run_standard::run_standard_chunk_loop;

/// Shared per-chunk InnerQ poll used by every prefill path (standard /
/// batched-prefill / batched-mixed). `maybe_finalize` is idempotent post
/// activation, and a no-op when `TURBO_INNERQ` was not set at startup —
/// so calling on every chunk costs one OnceLock load in the disabled case.
/// On non-cuda backends the driver doesn't exist (it talks to the CUDA
/// Driver API directly via `atlas_core::registry`), so this collapses to
/// a no-op via the `#[cfg]` gate.
#[cfg(feature = "cuda")]
pub(super) fn poll_innerq() {
    if let Some(driver) = spark_model::layers::qwen3_attention::INNERQ.get()
        && let Err(e) = driver.maybe_finalize(128)
    {
        tracing::warn!("InnerQ maybe_finalize failed: {e:#}");
    }
}

#[cfg(not(feature = "cuda"))]
pub(super) fn poll_innerq() {}

fn q12_dispatch_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("ATLAS_BISECT_Q12_DISABLE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn continue_in_progress_prefills(
    model: &dyn Model,
    policy: &dyn SchedulingPolicy,
    active: &mut Vec<ActiveSeq>,
    prefilling: &mut Vec<PrefillInProgress>,
    max_prefill_tokens: usize,
    chunk_plan: super::prefill_chunk_policy::ChunkPlan,
    prefill_stream: u64,
    prefill_event: u64,
    use_mtp: bool,
    use_self_speculative: bool,
    use_ngram_speculative: bool,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    adaptive_sampling: bool,
    ledger: &mut super::prefill_slo::PrefillTickLedger,
) -> bool {
    let mut did_mixed_step = false;

    // A peer may disappear, or its server deadline may expire, while a prior
    // GPU chunk is in flight. Retire such requests before launching another
    // chunk; `free_sequence` drains DFlash async state before freeing buffers.
    retire_stopped_prefills(model, prefilling, prefill_stream);

    if prefilling.is_empty() {
        return did_mixed_step;
    }

    // Check policy: skip chunks if active sequences are near TBT deadline.
    let timings: Vec<ActiveSeqTiming> = active
        .iter()
        .map(|a| ActiveSeqTiming {
            last_token_time: a.last_token_time,
        })
        .collect();
    let do_chunks = active.is_empty() || policy.should_prefill(&timings);

    if !do_chunks {
        return did_mixed_step;
    }

    let mut completed_indices = Vec::new();

    // Q12 batched-prefill paths. Two branches fire when 2+ streams are
    // prefilling concurrently (replaces the FIFO `prefilling.first_mut()`
    // advance — see qwen-refactor notes §6 for the asymmetric-TTFT
    // bug it fixes). The active-empty case routes to `prefill_batch_chunk`;
    // active-nonempty routes to `mixed_forward_batch` (N decode + M
    // prefill fused). Both call the default trait impl today (per-stream
    // loops); Q12 Phase 2/3 replace with kernel-level batched dispatch.
    //
    // Gates: N≥2 prefilling, no EP (worker opcode pending, Phase 6),
    // and for mixed-batch: skip whenever any speculative path is enabled
    // (matching `run_standard.rs:82-84`). Even at `active.len() >= 2` on
    // non-mHC DFlash/MTP models (`hc_mult() == 0`), `mixed_forward_batch`
    // omits DFlash `ctx_hidden_acc` capture and leaves stale `pending_drafts`
    // on sequences carried over from `active.len() == 1`.
    let spec_active = use_mtp || use_self_speculative || use_ngram_speculative;
    // BISECT: ATLAS_BISECT_Q12_DISABLE=1 forces the per-stream FIFO path
    // (pre-Q12 behavior) so we can isolate whether the chunked-prefill +
    // concurrent-decode crash originates in the Q12 batched-prefill
    // dispatch or pre-existing chunked-prefill state mutation.
    let q12_dispatch_disabled = q12_dispatch_disabled();
    // mHC/Qwen4 prefill owns slot-local PLE and highway state, while the Q12
    // batched path still uses ordinary residual semantics.  Advance one
    // prompt at a time until that path has explicit highway support.
    let serialize_hc = model.hc_mult() > 0;
    let can_batch_prefill_only = !q12_dispatch_disabled
        && !serialize_hc
        && prefilling.len() >= 2
        && active.is_empty()
        && !model.is_ep();
    let can_batch_mixed = can_batch_mixed_step(
        q12_dispatch_disabled,
        serialize_hc,
        prefilling.len(),
        active.len(),
        spec_active,
        model.is_ep(),
    );

    if can_batch_prefill_only {
        ledger.taint();
        run_batched_prefill_step(
            model,
            prefilling,
            &mut completed_indices,
            max_prefill_tokens,
            prefill_stream,
            prefill_event,
            think_end_token,
        );
        promote_completed_prefills(
            model,
            prefilling,
            completed_indices,
            active,
            prefill_stream,
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
        );
        return did_mixed_step;
    }

    if can_batch_mixed {
        ledger.taint();
        let t0_mixed = Instant::now();
        run_batched_mixed_step(
            model,
            active,
            prefilling,
            &mut completed_indices,
            max_prefill_tokens,
            prefill_stream,
            prefill_event,
            t0_mixed,
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            reflection_suppress_ids,
            adaptive_sampling,
            &mut did_mixed_step,
        );
        promote_completed_prefills(
            model,
            prefilling,
            completed_indices,
            active,
            prefill_stream,
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
        );
        return did_mixed_step;
    }

    // When no active decode sequences exist, run all remaining chunks across
    // all in-progress prefills in a tight loop to minimize TTFT and promote
    // multi-chunk cohorts together. Otherwise, run 1 chunk on the SRPT-selected
    // (or first FIFO) in-progress prefill and yield to decode.
    let process_indices: Vec<usize> = if active.is_empty() {
        (0..prefilling.len()).collect()
    } else if prefilling.len() > 1 {
        let srpt_cfg =
            super::prefill_slo::prefill_srpt::SrptConfig::from_env(chunk_plan.widths.is_some());
        vec![select_active_continue_index(prefilling, srpt_cfg)]
    } else {
        vec![0]
    };
    let mut retired_indices = Vec::new();
    for idx in process_indices {
        let p = &mut prefilling[idx];
        let mut two_phase_failed = false;

        // Two-phase SSM prefill: when the full sequence hasn't started
        // chunking yet (chunk_offset == 0) and is longer than one chunk,
        // use the two-phase path for better SSM state quality.
        let use_twophase = crate::prefill_twophase_gate::should_use_twophase(
            model.supports_prefill_twophase(),
            p.chunk_offset,
            p.prompt_tokens.len(),
            max_prefill_tokens,
        );
        if use_twophase {
            if stop_allocated_prefill(
                model,
                prefill_stream,
                &mut p.seq,
                &mut p.reset_proof,
                &mut p.sink,
                p.cancel_flag.as_ref(),
                p.timeout_at,
                "two-phase prefill continuation",
            ) {
                retired_indices.push(idx);
                continue;
            }
            ledger.taint();
            tracing::info!(
                "Two-phase prefill: {} tokens, chunk_size={}",
                p.prompt_tokens.len(),
                max_prefill_tokens,
            );
            match model.prefill_twophase(
                &p.prompt_tokens,
                &mut p.seq,
                max_prefill_tokens,
                prefill_stream,
            ) {
                Ok(logits) => {
                    let chunk_tokens = p.prompt_tokens.len().saturating_sub(p.chunk_offset);
                    p.chunk_offset = p.prompt_tokens.len();
                    crate::metrics::record_prefill_progress(
                        p.session_hash,
                        p.chunk_offset,
                        p.prompt_tokens.len(),
                        chunk_tokens,
                    );
                    complete_prefill_handoff(
                        model,
                        prefill_stream,
                        prefill_event,
                        "two-phase prefill handoff",
                    );
                    let params = p.sampling_params();
                    let apply_grammar =
                        prefill_first_token_applies_grammar(p.enable_thinking, think_end_token);
                    match sample_token_with_logprobs_and_grammar(
                        model,
                        logits,
                        &params,
                        &p.eos_tokens,
                        &[],
                        p.top_logprobs,
                        p.grammar_state.as_mut(),
                        apply_grammar,
                    ) {
                        Ok((first, first_lp)) => {
                            super::logits_row_receipt::prefill_raw(
                                model,
                                logits,
                                p.session_hash,
                                first,
                            );
                            p.first_logprobs = first_lp;
                            tracing::info!("Two-phase prefill first token: {first}");
                            completed_indices.push((idx, Some(first)));
                        }
                        Err(e) => {
                            drain_before_prefill_cleanup(
                                model,
                                prefill_stream,
                                "two-phase prefill sampling failure",
                            );
                            tracing::error!("Two-phase prefill sampling: {e:#}");
                            completed_indices.push((idx, None));
                        }
                    }
                }
                Err(e) => {
                    drain_before_prefill_cleanup(
                        model,
                        prefill_stream,
                        "two-phase prefill failure",
                    );
                    tracing::error!(
                        "Two-phase prefill failed; request will not be replayed: {e:#}"
                    );
                    completed_indices.push((idx, None));
                    two_phase_failed = true;
                }
            }
        }

        // A failed two-phase attempt may have partially mutated recurrent/KV
        // state. It is request-fatal and must never be replayed as chunked
        // prefill against the same sequence.
        if !two_phase_failed && p.chunk_offset < p.prompt_tokens.len() {
            loop {
                if stop_allocated_prefill(
                    model,
                    prefill_stream,
                    &mut p.seq,
                    &mut p.reset_proof,
                    &mut p.sink,
                    p.cancel_flag.as_ref(),
                    p.timeout_at,
                    "chunked-prefill continuation",
                ) {
                    retired_indices.push(idx);
                    break;
                }
                let prev_offset = p.chunk_offset;
                let prev_completed = completed_indices.len();
                run_standard_chunk_loop(
                    model,
                    p,
                    idx,
                    active,
                    max_prefill_tokens,
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
                    reflection_suppress_ids,
                    adaptive_sampling,
                    &mut completed_indices,
                    &mut did_mixed_step,
                    ledger,
                );
                if !active.is_empty()
                    || chunk_plan.slo.is_some()
                    || p.chunk_offset >= p.prompt_tokens.len()
                    || completed_indices.len() > prev_completed
                    || p.chunk_offset == prev_offset
                {
                    break;
                }
            }
        }
    }

    remove_retired_prefills(prefilling, &mut completed_indices, &mut retired_indices);

    // Move completed prefills to active (or free on error).
    promote_completed_prefills(
        model,
        prefilling,
        completed_indices,
        active,
        prefill_stream,
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
    );

    did_mixed_step
}

/// Drop entries retired inside the continuation loop while keeping every index
/// in `completed_indices` valid and in its original FIFO order.
fn remove_retired_prefills(
    prefilling: &mut Vec<PrefillInProgress>,
    completed_indices: &mut [(usize, Option<u32>)],
    retired_indices: &mut Vec<usize>,
) {
    if retired_indices.is_empty() {
        return;
    }
    retired_indices.sort_unstable_by_key(|&idx| std::cmp::Reverse(idx));
    retired_indices.dedup();
    for retired_idx in retired_indices.drain(..) {
        debug_assert!(
            completed_indices.iter().all(|(idx, _)| *idx != retired_idx),
            "retired prefill index {retired_idx} must not also be queued for promotion"
        );
        let _ = prefilling.remove(retired_idx);
        for (completed_idx, _) in completed_indices.iter_mut() {
            if *completed_idx > retired_idx {
                *completed_idx -= 1;
            }
        }
    }
}

/// Pick the single in-progress prefill index to advance while decode rows are
/// active: shortest effective remaining tokens under SRPT, or index `0` under FIFO.
fn select_active_continue_index(
    prefilling: &[PrefillInProgress],
    srpt_cfg: super::prefill_slo::prefill_srpt::SrptConfig,
) -> usize {
    if !srpt_cfg.enabled || prefilling.len() <= 1 {
        return 0;
    }
    let candidates: Vec<(u64, usize, usize)> = prefilling
        .iter()
        .map(|p| {
            let key =
                super::prefill_slo::prefill_srpt::prefill_wait_key(p.session_hash, p.seq.slot_idx);
            let remaining = p.prompt_tokens.len().saturating_sub(p.chunk_offset);
            (key, remaining, p.chunk_offset)
        })
        .collect();
    super::prefill_slo::prefill_srpt::select_in_flight_index(srpt_cfg, &candidates).unwrap_or(0)
}

pub(super) const fn can_batch_mixed_step(
    q12_dispatch_disabled: bool,
    serialize_hc: bool,
    prefilling_len: usize,
    active_len: usize,
    spec_active: bool,
    is_ep: bool,
) -> bool {
    !q12_dispatch_disabled
        && !serialize_hc
        && prefilling_len >= 2
        && active_len > 0
        && !spec_active
        && !is_ep
}

#[cfg(test)]
mod tests {
    use super::{can_batch_mixed_step, continue_in_progress_prefills, remove_retired_prefills};
    use crate::scheduler::prefill_chunk_policy::ChunkPlan;
    use crate::scheduler::prefill_slo::PrefillTickLedger;
    use crate::scheduler::test_fixtures::{model, prefill_in_progress};
    use crate::scheduler::types::ResponseSink;
    use crate::scheduling_policy::FifoPolicy;

    #[test]
    fn can_batch_mixed_step_blocks_speculative_decode_at_all_active_widths() {
        for active_len in [1usize, 2, 4, 8] {
            assert!(
                !can_batch_mixed_step(false, false, 2, active_len, true, false),
                "spec_active must bar mixed_forward_batch at active_len={active_len}"
            );
            assert!(
                can_batch_mixed_step(false, false, 2, active_len, false, false),
                "non-spec non-HC non-EP batch should admit active_len={active_len}"
            );
        }
        assert!(!can_batch_mixed_step(false, false, 1, 2, false, false));
        assert!(!can_batch_mixed_step(false, false, 2, 0, false, false));
        assert!(!can_batch_mixed_step(true, false, 2, 2, false, false));
        assert!(!can_batch_mixed_step(false, true, 2, 2, false, false));
        assert!(!can_batch_mixed_step(false, false, 2, 2, false, true));
    }

    #[test]
    fn continuation_tight_loop_retires_disconnected_prefill_before_next_chunk() {
        let m = model::recording();
        *m.prefill_chunks.lock().expect("prefill_chunks lock") = Some(Vec::new());
        let (tx, rx) = tokio::sync::oneshot::channel();
        *m.disconnect_after_chunks
            .lock()
            .expect("disconnect_after_chunks lock") = Some((1, Box::new(rx)));

        let mut p = prefill_in_progress(ResponseSink::Blocking(Some(tx)));
        p.prompt_tokens = vec![1; 64];
        p.chunk_offset = 16;
        p.chunk_budget = 16;
        let mut prefilling = vec![p];
        let mut active = Vec::new();
        let mut ledger = PrefillTickLedger::default();

        let did_mixed = continue_in_progress_prefills(
            &m,
            &FifoPolicy,
            &mut active,
            &mut prefilling,
            16,
            ChunkPlan::default(),
            7,
            8,
            false,
            false,
            false,
            None,
            None,
            None,
            None,
            None,
            &[],
            false,
            &mut ledger,
        );

        assert!(!did_mixed);
        assert!(prefilling.is_empty(), "retired prefill must be removed");
        assert!(active.is_empty(), "retired prefill must not be promoted");
        assert_eq!(
            *m.prefill_chunks.lock().expect("prefill_chunks lock"),
            Some(vec![(16, 16)]),
            "must stop before launching chunk 2 once disconnected"
        );
        assert_eq!(
            *m.lifecycle_events.lock().expect("lifecycle_events lock"),
            ["sync:7", "sync:0", "free:0"],
            "must drain prefill and default streams before freeing sequence"
        );
    }

    #[test]
    fn remove_retired_prefills_keeps_completed_indices_aligned_and_fifo() {
        let mut prefilling: Vec<_> = (10u64..14)
            .map(|hash| {
                let mut p = prefill_in_progress(ResponseSink::Blocking(None));
                p.session_hash = hash;
                p
            })
            .collect();
        let mut completed = vec![(0, Some(100)), (2, Some(200))];
        let mut retired = vec![3, 1, 1];

        remove_retired_prefills(&mut prefilling, &mut completed, &mut retired);

        assert!(retired.is_empty());
        assert_eq!(completed, vec![(0, Some(100)), (1, Some(200))]);
        let hashes: Vec<u64> = prefilling.iter().map(|p| p.session_hash).collect();
        assert_eq!(hashes, vec![10, 12]);
    }

    #[test]
    fn srpt_continuation_picks_shortest_remaining_in_flight_while_decoding() {
        use super::select_active_continue_index;
        use crate::scheduler::prefill_chunk_policy::PreparedWidths;
        use crate::scheduler::prefill_slo::prefill_srpt::SrptConfig;
        use crate::scheduler::test_fixtures::active_seq;

        let mut long_p = prefill_in_progress(ResponseSink::Blocking(None));
        long_p.prompt_tokens = vec![1; 8_192];
        long_p.chunk_offset = 512;
        long_p.session_hash = 0x1001;
        long_p.seq.slot_idx = 1;

        let mut short_p = prefill_in_progress(ResponseSink::Blocking(None));
        short_p.prompt_tokens = vec![2; 1_536];
        short_p.chunk_offset = 512;
        short_p.session_hash = 0x2002;
        short_p.seq.slot_idx = 2;

        let mut prefilling = vec![long_p, short_p];
        let fifo_cfg = SrptConfig {
            enabled: false,
            ..SrptConfig::default()
        };
        assert_eq!(
            select_active_continue_index(&prefilling, fifo_cfg),
            0,
            "FIFO mode must keep index 0"
        );
        assert_eq!(
            select_active_continue_index(&prefilling, SrptConfig::default()),
            1,
            "SRPT must pick the shorter remaining prefill at index 1"
        );

        let m = model::recording();
        *m.prefill_chunks.lock().expect("prefill_chunks lock") = Some(Vec::new());
        let mut active = vec![active_seq(ResponseSink::Blocking(None), 0)];
        let mut ledger = PrefillTickLedger::default();
        let chunk_plan = ChunkPlan {
            widths: PreparedWidths::new(&[512, 1024, 2048, 4096]),
            fallback_width: 512,
            carry_align: 16,
            ..ChunkPlan::default()
        };

        let did_mixed = continue_in_progress_prefills(
            &m,
            &FifoPolicy,
            &mut active,
            &mut prefilling,
            4096,
            chunk_plan,
            7,
            8,
            true, // spec_active disables Q12 mixed_forward_batch, matching native MTP
            false,
            false,
            None,
            None,
            None,
            None,
            None,
            &[],
            false,
            &mut ledger,
        );

        assert!(!did_mixed);
        assert_eq!(
            ledger.chunks(),
            1,
            "at most 1 chunk per tick while decoding"
        );
        if SrptConfig::from_env(true).enabled {
            assert_eq!(prefilling[0].chunk_offset, 512, "long prefill waits");
            assert_eq!(
                prefilling[1].chunk_offset, 1024,
                "short prefill advances first under SRPT"
            );
        }
    }
}
