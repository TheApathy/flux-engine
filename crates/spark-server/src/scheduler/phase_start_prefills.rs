// SPDX-License-Identifier: AGPL-3.0-only

//! Phase: start new requests — either single-shot prefill (legacy), batched
//! Qwen4 short-prefill bursts (`2..=16` single-chunk prompts sharing one
//! unified MoE pass), or serial chunked prefill that pushes onto `prefilling`.
//! A start failure is logged only: the request's sink has already been
//! answered by the prefill entry point, and active sequences are never
//! preempted for it.

use std::collections::VecDeque;

use spark_model::traits::{Model, PrefillSlice};

use super::prefill_stream_barrier::{complete_prefill_handoff, drain_before_prefill_cleanup};
use super::*;
use crate::api::InferenceRequest;
use crate::grammar::GrammarEngine;

/// Runtime kill-switch for Qwen4 batched short-prefill (default ON; `0` disables).
const SHORT_BATCH_ENV: &str = "ATLAS_QWEN4_PREFILL_SHORT_BATCH";

/// Minimum concurrent prompts required to invoke batched short-prefill.
const SHORT_BATCH_MIN_STREAMS: usize = 2;

/// Maximum concurrent prompts collected into one batched short-prefill round.
const SHORT_BATCH_MAX_STREAMS: usize = 16;

/// Hard ceiling on total packed prompt rows across all streams in one round.
const SHORT_BATCH_MAX_ROWS: usize = 2048;

/// Parse `ATLAS_QWEN4_PREFILL_SHORT_BATCH`: default ON, explicit `"0"` disables.
#[must_use]
fn parse_short_batch_enabled(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("0")
}

#[must_use]
fn short_batch_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED
        .get_or_init(|| parse_short_batch_enabled(std::env::var(SHORT_BATCH_ENV).ok().as_deref()))
}

#[must_use]
fn is_qwen4_short_batch_model(model: &dyn Model) -> bool {
    model.hc_mult() > 0 && !model.is_ep() && matches!(model.qwen4_batch_plan(), Ok(Some(_)))
}

#[must_use]
fn request_has_images(req: &InferenceRequest) -> bool {
    match req {
        InferenceRequest::Blocking { image_pixels, .. }
        | InferenceRequest::Streaming { image_pixels, .. } => !image_pixels.is_empty(),
    }
}

/// Pure metadata summary for one queued request considered by the short-batch collector.
#[derive(Clone, Copy, Debug)]
struct ShortBatchCandidate {
    prompt_len: usize,
    single_chunk_budget: usize,
    has_images: bool,
    has_disagg: bool,
}

/// Hard capacity limits governing one batched short-prefill round.
#[derive(Clone, Copy, Debug)]
struct ShortBatchLimits {
    max_streams: usize,
    max_total_rows: usize,
    block_size: usize,
    avail_kv_blocks: Option<usize>,
}

/// Greedily collect a strict FIFO prefix of `2..=16` eligible short single-chunk
/// requests whose combined prompt length fits `<= min(max_total_rows, 2048)`
/// and available batch-slot / KV-block budgets. Returns `0` when fewer than 2
/// front requests qualify so the caller falls through to the serial path.
#[must_use]
fn select_short_batch_prefix(
    candidates: &[ShortBatchCandidate],
    limits: ShortBatchLimits,
) -> usize {
    let stream_cap = limits.max_streams.min(SHORT_BATCH_MAX_STREAMS);
    if stream_cap < SHORT_BATCH_MIN_STREAMS || candidates.len() < SHORT_BATCH_MIN_STREAMS {
        return 0;
    }
    let row_cap = limits.max_total_rows.min(SHORT_BATCH_MAX_ROWS);
    let block_size = limits.block_size.max(1);

    let mut count = 0usize;
    let mut total_rows = 0usize;
    let mut used_blocks = 0usize;

    for cand in candidates {
        if count >= stream_cap {
            break;
        }
        if cand.has_images
            || cand.has_disagg
            || cand.prompt_len < 2
            || cand.prompt_len > cand.single_chunk_budget
        {
            break;
        }
        let Some(next_rows) = total_rows.checked_add(cand.prompt_len) else {
            break;
        };
        if next_rows > row_cap {
            break;
        }
        if let Some(free_blocks) = limits.avail_kv_blocks {
            let req_blocks = cand.prompt_len.div_ceil(block_size);
            let Some(next_blocks) = used_blocks.checked_add(req_blocks) else {
                break;
            };
            if next_blocks > free_blocks {
                break;
            }
            used_blocks = next_blocks;
        }
        total_rows = next_rows;
        count += 1;
    }

    if count >= SHORT_BATCH_MIN_STREAMS {
        count
    } else {
        0
    }
}

struct ClaimedShortBatchItem {
    prep: PreparedPrefillRequest,
    seq: SequenceState,
    reset_proof: Option<super::continuation::ResetProofPayload>,
    first_start: usize,
    first_len: usize,
}

#[allow(clippy::too_many_arguments)]
fn drain_pending_for_burst(
    model: &dyn Model,
    pending: &std::sync::Arc<(Mutex<PendingQueue>, Condvar)>,
    req_queue: &mut VecDeque<InferenceRequest>,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    swapped: &[SwappedSeq],
    admit_watermark: usize,
    max_seq_len: usize,
    block_size: usize,
    verify_lookahead: usize,
    max_batch_size: usize,
    defer_first_stream_event: &mut bool,
    mid_loop_kv_full: &mut bool,
) {
    if *mid_loop_kv_full || active.len() + prefilling.len() + req_queue.len() >= max_batch_size {
        return;
    }
    let drained = {
        let mut g = pending.0.lock();
        let cap = max_batch_size.saturating_sub(active.len() + prefilling.len() + req_queue.len());
        let take_n = cap.min(g.requests.len());
        if take_n > 0 {
            let v: Vec<InferenceRequest> = g.requests.drain(..take_n).collect();
            pending.1.notify_all();
            v
        } else {
            Vec::new()
        }
    };
    if drained.is_empty() {
        return;
    }
    let reserved_demands: Vec<super::admission::SeqDemand> = req_queue
        .iter()
        .map(|queued| super::admission::SeqDemand {
            current_tokens: queued.prompt_len(),
            budget_tokens: queued.max_tokens(),
        })
        .collect();
    let admitted = super::admission::gate_admissions_with_reserved(
        model,
        pending,
        drained,
        active,
        prefilling,
        swapped,
        &reserved_demands,
        admit_watermark,
        max_seq_len,
        block_size,
        verify_lookahead,
    );
    if !admitted.is_empty() {
        *defer_first_stream_event = true;
        req_queue.extend(admitted);
    } else {
        *mid_loop_kv_full = true;
    }
}

#[allow(clippy::too_many_arguments)]
fn run_qwen4_short_prefill_bursts(
    model: &dyn Model,
    req_queue: &mut VecDeque<InferenceRequest>,
    pending: &std::sync::Arc<(Mutex<PendingQueue>, Condvar)>,
    swapped: &[SwappedSeq],
    admit_watermark: usize,
    max_seq_len: usize,
    block_size: usize,
    verify_lookahead: usize,
    max_batch_size: usize,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    chunk_plan: super::prefill_chunk_policy::ChunkPlan,
    eos_tokens: &[u32],
    prefill_stream: u64,
    prefill_event: u64,
    grammar_engine: &mut Option<GrammarEngine>,
    tokens_cfg: PrefillTokenConfig,
    complete_all_chunks: bool,
    defer_first_stream_event: &mut bool,
    mid_loop_kv_full: &mut bool,
    active: &mut Vec<ActiveSeq>,
    prefilling: &mut Vec<PrefillInProgress>,
    ledger: &mut super::prefill_slo::PrefillTickLedger,
) {
    loop {
        if complete_all_chunks {
            drain_pending_for_burst(
                model,
                pending,
                req_queue,
                active,
                prefilling,
                swapped,
                admit_watermark,
                max_seq_len,
                block_size,
                verify_lookahead,
                max_batch_size,
                defer_first_stream_event,
                mid_loop_kv_full,
            );
        }
        if req_queue.len() < SHORT_BATCH_MIN_STREAMS {
            break;
        }

        let active_decode_interleave = !complete_all_chunks && !active.is_empty();
        let candidates: Vec<ShortBatchCandidate> = req_queue
            .iter()
            .map(|req| {
                let prompt_len = req.prompt_len();
                let single_chunk_budget = super::prefill_chunk_policy::initial_chunk_budget(
                    max_prefill_tokens,
                    max_batch_tokens,
                    chunk_plan,
                    active_decode_interleave,
                    model.is_mla(),
                    prompt_len,
                )
                .min(max_prefill_tokens);
                ShortBatchCandidate {
                    prompt_len,
                    single_chunk_budget,
                    has_images: request_has_images(req),
                    has_disagg: crate::api::disagg::has_pending_prefill_export(req.session_hash()),
                }
            })
            .collect();

        let limits = ShortBatchLimits {
            max_streams: max_batch_size.saturating_sub(active.len() + prefilling.len()),
            max_total_rows: SHORT_BATCH_MAX_ROWS.min(max_batch_tokens),
            block_size,
            avail_kv_blocks: (model.num_total_blocks() > 0 && block_size > 0)
                .then(|| model.num_free_blocks()),
        };

        let batch_n = select_short_batch_prefix(&candidates, limits);
        if batch_n < SHORT_BATCH_MIN_STREAMS {
            break;
        }

        let mut batch_reqs: VecDeque<InferenceRequest> = VecDeque::with_capacity(batch_n);
        for _ in 0..batch_n {
            if let Some(req) = req_queue.pop_front() {
                batch_reqs.push_back(req);
            }
        }
        *defer_first_stream_event = true;

        // Prepare and claim sequence slots using the exact same lifecycle and
        // admission checks as single-request chunked prefill.
        let mut claimed: Vec<ClaimedShortBatchItem> = Vec::with_capacity(batch_reqs.len());
        let mut hit_claim_error = false;
        while let Some(req) = batch_reqs.pop_front() {
            let mut prep = match prepare_prefill_request(req, eos_tokens, grammar_engine) {
                Ok(p) => p,
                Err(e) => {
                    ledger.taint();
                    handle_prefill_start_error(&e, active.len());
                    continue;
                }
            };
            let total = prep.prompt_tokens.len();
            if stop_unallocated_prefill(
                prep.cancel_flag.as_ref(),
                prep.timeout_at,
                &mut prep.sink,
                "short-batch",
            ) {
                ledger.record_chunk(total, 0);
                continue;
            }
            if claimed.is_empty() {
                model.begin_prefill_admission(&prep.prompt_tokens, total);
            }
            let (mut seq, reset_proof) =
                match super::continuation::ContinuationRecord::claim_sequence_on(
                    model,
                    prefill_stream,
                    prep.prefix_hash,
                ) {
                    Ok(pair) => pair,
                    Err(e) => {
                        let msg = format!("alloc_sequence failed: {e:#}");
                        send_error_to_sink(&mut prep.sink, &msg);
                        ledger.taint();
                        handle_prefill_start_error(&e, active.len());
                        // Restore remaining unclaimed requests to the front of
                        // `req_queue` in strict FIFO order for serial fallback.
                        while let Some(unclaimed) = batch_reqs.pop_back() {
                            req_queue.push_front(unclaimed);
                        }
                        hit_claim_error = true;
                        break;
                    }
                };
            seq.session_hash = prep.session_hash;
            seq.qwen4_qsa_required = prefill_request_can_reach_qwen4_qsa(total, prep.max_tokens);
            super::qwen4_batch_route::record_request_admission(model, &seq, total, prep.max_tokens);
            claimed.push(ClaimedShortBatchItem {
                prep,
                seq,
                reset_proof,
                first_start: 0,
                first_len: total,
            });
        }

        if claimed.is_empty() {
            if hit_claim_error {
                break;
            }
            continue;
        }

        let total_batch_tokens: usize = claimed.iter().map(|c| c.prep.prompt_tokens.len()).sum();
        tracing::info!(
            "Qwen4 short-batch prefill start: streams={}, total_tokens={}",
            claimed.len(),
            total_batch_tokens,
        );

        let prefill_result = (|| -> Result<Vec<DevicePtr>> {
            // Invalidate any prior request's published images (text-only batch).
            model.prepare_vision_embed(&[])?;

            for item in &mut claimed {
                let total = item.prep.prompt_tokens.len();
                let first_start =
                    model.resume_conversation(&mut item.seq, &item.prep.prompt_tokens)?;
                anyhow::ensure!(
                    first_start < total,
                    "resume_conversation offset {first_start} >= prompt len {total}"
                );
                item.first_start = first_start;
                item.first_len = total - first_start;
            }

            if claimed.len() == 1 {
                let item = &mut claimed[0];
                let logits = model.prefill_chunk(
                    &item.prep.prompt_tokens,
                    &mut item.seq,
                    item.first_start,
                    item.first_len,
                    true,
                    prefill_stream,
                )?;
                return Ok(vec![logits]);
            }

            let mut slices: Vec<PrefillSlice<'_>> = claimed
                .iter_mut()
                .map(|item| PrefillSlice {
                    prompt_tokens: &item.prep.prompt_tokens,
                    seq: &mut item.seq,
                    chunk_start: item.first_start,
                    chunk_len: item.first_len,
                    is_last_chunk: true,
                })
                .collect();
            let logits = model.prefill_batch_chunk(&mut slices, prefill_stream)?;
            anyhow::ensure!(
                logits.len() == claimed.len(),
                "prefill_batch_chunk returned {} logits for {} short-batch streams",
                logits.len(),
                claimed.len()
            );
            Ok(logits)
        })();

        let logits_ptrs = match prefill_result {
            Ok(ptrs) => ptrs,
            Err(e) => {
                drain_before_prefill_cleanup(model, prefill_stream, "short-batch prefill failure");
                let msg = format!("prefill_batch_chunk failed: {e:#}");
                for item in &mut claimed {
                    send_error_to_sink(&mut item.prep.sink, &msg);
                    if let Err(free_err) =
                        super::continuation::ContinuationRecord::abort_prefill_sequence(
                            model,
                            &mut item.seq,
                            &mut item.reset_proof,
                        )
                    {
                        tracing::error!(
                            "phase_start_prefills: free_sequence (after short-batch error): {free_err:#}"
                        );
                    }
                    if let Err(bcast_err) = model.ep_broadcast_cmd(0xFFFFFFF1) {
                        tracing::error!(
                            "phase_start_prefills: ep_broadcast (after short-batch error): {bcast_err:#}"
                        );
                    }
                }
                ledger.taint();
                handle_prefill_start_error(&e, active.len());
                break;
            }
        };

        if let Some(next_req) = req_queue.front() {
            let next_tokens = next_req.prompt_tokens();
            let next_chunk_len = next_tokens.len().min(max_prefill_tokens);
            model.begin_prefill_admission(next_tokens, next_chunk_len);
        }
        complete_prefill_handoff(model, prefill_stream, prefill_event, "short-batch handoff");

        for (mut item, logits) in claimed.into_iter().zip(logits_ptrs) {
            let prompt_len = item.prep.prompt_tokens.len();
            crate::metrics::record_prefill_progress(
                item.prep.session_hash,
                prompt_len,
                prompt_len,
                item.first_len,
            );
            if stop_allocated_prefill(
                model,
                prefill_stream,
                &mut item.seq,
                &mut item.reset_proof,
                &mut item.prep.sink,
                item.prep.cancel_flag.as_ref(),
                item.prep.timeout_at,
                "short-batch",
            ) {
                ledger.record_chunk(prompt_len, 0);
                continue;
            }
            match finalize_prefill_first_token(
                tokens_cfg,
                model,
                item.prep,
                item.seq,
                item.reset_proof,
                logits,
                None,
                false,
                prefill_stream,
                *defer_first_stream_event,
            ) {
                Ok(StartPrefillResult::Active(a)) => {
                    ledger.record_chunk(prompt_len, 0);
                    tracing::info!(
                        "Prefilled (short-batch): seq_len={}, remaining={}",
                        a.seq.seq_len,
                        a.remaining,
                    );
                    active.push(a);
                }
                Ok(StartPrefillResult::Finished) => {
                    ledger.record_chunk(prompt_len, 0);
                }
                Ok(StartPrefillResult::InProgress(p)) => {
                    ledger.record_chunk(p.chunk_offset, 0);
                    prefilling.push(p);
                }
                Err(e) => {
                    ledger.taint();
                    handle_prefill_start_error(&e, active.len());
                }
            }
        }

        if hit_claim_error {
            break;
        }
    }
}

#[cfg(test)]
fn initial_prefill_chunk_budget(max_prefill_tokens: usize, max_batch_tokens: usize) -> usize {
    super::prefill_chunk_policy::initial_chunk_budget(
        max_prefill_tokens,
        max_batch_tokens,
        super::prefill_chunk_policy::ChunkPlan::default(),
        false,
        false,
        max_prefill_tokens,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn start_new_requests(
    model: &dyn Model,
    new_reqs: Vec<InferenceRequest>,
    pending: &std::sync::Arc<(Mutex<PendingQueue>, Condvar)>,
    swapped: &[SwappedSeq],
    admit_watermark: usize,
    max_seq_len: usize,
    block_size: usize,
    verify_lookahead: usize,
    max_batch_size: usize,
    chunked: bool,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    chunk_plan: super::prefill_chunk_policy::ChunkPlan,
    eos_tokens: &[u32],
    prefill_stream: u64,
    prefill_event: u64,
    grammar_engine: &mut Option<GrammarEngine>,
    spontaneous_think_budget: u32,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    active: &mut Vec<ActiveSeq>,
    prefilling: &mut Vec<PrefillInProgress>,
    ledger: &mut super::prefill_slo::PrefillTickLedger,
) {
    start_new_requests_with_short_batch(
        model,
        new_reqs,
        pending,
        swapped,
        admit_watermark,
        max_seq_len,
        block_size,
        verify_lookahead,
        max_batch_size,
        chunked,
        max_prefill_tokens,
        max_batch_tokens,
        chunk_plan,
        eos_tokens,
        prefill_stream,
        prefill_event,
        grammar_engine,
        spontaneous_think_budget,
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        active,
        prefilling,
        ledger,
        short_batch_enabled(),
    );
}

#[allow(clippy::too_many_arguments)]
fn start_new_requests_with_short_batch(
    model: &dyn Model,
    new_reqs: Vec<InferenceRequest>,
    pending: &std::sync::Arc<(Mutex<PendingQueue>, Condvar)>,
    swapped: &[SwappedSeq],
    admit_watermark: usize,
    max_seq_len: usize,
    block_size: usize,
    verify_lookahead: usize,
    max_batch_size: usize,
    chunked: bool,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    chunk_plan: super::prefill_chunk_policy::ChunkPlan,
    eos_tokens: &[u32],
    prefill_stream: u64,
    prefill_event: u64,
    grammar_engine: &mut Option<GrammarEngine>,
    spontaneous_think_budget: u32,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    active: &mut Vec<ActiveSeq>,
    prefilling: &mut Vec<PrefillInProgress>,
    ledger: &mut super::prefill_slo::PrefillTickLedger,
    short_batch_allowed: bool,
) {
    let active_start = active.len();
    let complete_all_chunks = chunked
        && active_start == 0
        && prefilling.is_empty()
        && model.hc_mult() > 0
        && chunk_plan.slo.is_none();
    let mut req_queue: VecDeque<InferenceRequest> = new_reqs.into();
    let mut defer_first_stream_event = chunked && req_queue.len() > 1;
    let mut mid_loop_kv_full = false;

    if chunked
        && short_batch_allowed
        && !req_queue.is_empty()
        && is_qwen4_short_batch_model(model)
        && crate::api::disagg::current_runtime_config().role
            == atlas_core::config::DisaggRole::Unified
    {
        let tokens_cfg = PrefillTokenConfig {
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            spontaneous_think_budget,
        };
        run_qwen4_short_prefill_bursts(
            model,
            &mut req_queue,
            pending,
            swapped,
            admit_watermark,
            max_seq_len,
            block_size,
            verify_lookahead,
            max_batch_size,
            max_prefill_tokens,
            max_batch_tokens,
            chunk_plan,
            eos_tokens,
            prefill_stream,
            prefill_event,
            grammar_engine,
            tokens_cfg,
            complete_all_chunks,
            &mut defer_first_stream_event,
            &mut mid_loop_kv_full,
            active,
            prefilling,
            ledger,
        );
    }

    while let Some(req) = req_queue.pop_front() {
        let prompt_len = req.prompt_len();
        if complete_all_chunks
            && !mid_loop_kv_full
            && active.len() + prefilling.len() + 1 + req_queue.len() < max_batch_size
        {
            let drained = {
                let mut g = pending.0.lock();
                let cap = max_batch_size
                    .saturating_sub(active.len() + prefilling.len() + 1 + req_queue.len());
                let take_n = cap.min(g.requests.len());
                if take_n > 0 {
                    let v: Vec<InferenceRequest> = g.requests.drain(..take_n).collect();
                    pending.1.notify_all();
                    v
                } else {
                    Vec::new()
                }
            };
            if !drained.is_empty() {
                let mut reserved_demands = Vec::with_capacity(1 + req_queue.len());
                reserved_demands.push(super::admission::SeqDemand {
                    current_tokens: prompt_len,
                    budget_tokens: req.max_tokens(),
                });
                reserved_demands.extend(req_queue.iter().map(|queued| {
                    super::admission::SeqDemand {
                        current_tokens: queued.prompt_len(),
                        budget_tokens: queued.max_tokens(),
                    }
                }));
                let admitted = super::admission::gate_admissions_with_reserved(
                    model,
                    pending,
                    drained,
                    active,
                    prefilling,
                    swapped,
                    &reserved_demands,
                    admit_watermark,
                    max_seq_len,
                    block_size,
                    verify_lookahead,
                );
                if !admitted.is_empty() {
                    defer_first_stream_event = true;
                    req_queue.extend(admitted);
                } else {
                    mid_loop_kv_full = true;
                }
            }
        }
        if chunked {
            // The arena includes extra decode-slot capacity, but that capacity
            // is not part of the explicit prefill ceiling. In particular, an
            // 8192-token kernel qualification must never become an 8193-token
            // launch merely because max_batch_size is one. Chunk 0 runs inside
            // this tick, before the active rows' decode step, so the bounded
            // interleave width (or the active-decode cap) applies to it too.
            // When `complete_all_chunks` is true (`active_start == 0`), all
            // burst prompts prefill back-to-back before the first decode step;
            // newly pushed burst rows in `active` are not awaiting decode
            // interleave, so keep full chunks instead of collapsing to 512.
            let budget = super::prefill_chunk_policy::initial_chunk_budget(
                max_prefill_tokens,
                max_batch_tokens,
                chunk_plan,
                !complete_all_chunks && !active.is_empty(),
                model.is_mla(),
                prompt_len,
            );
            let next_tokens = req_queue.front().map(|next_req| next_req.prompt_tokens());
            match start_chunked_prefill_with_defer(
                think_end_token,
                think_start_token,
                code_fence_token,
                tool_call_start_token,
                tool_call_end_token,
                model,
                req,
                eos_tokens,
                budget,
                prefill_stream,
                prefill_event,
                grammar_engine,
                spontaneous_think_budget,
                defer_first_stream_event,
                complete_all_chunks,
                next_tokens,
            ) {
                Ok(StartPrefillResult::Active(a)) => {
                    ledger.record_chunk(prompt_len, 0);
                    tracing::info!(
                        "Prefilled (single chunk): seq_len={}, remaining={}",
                        a.seq.seq_len,
                        a.remaining,
                    );
                    active.push(a);
                }
                Ok(StartPrefillResult::InProgress(p)) => {
                    ledger.record_chunk(p.chunk_offset, 0);
                    tracing::info!(
                        "Prefill chunk 0/{}: {}/{} tokens",
                        p.prompt_tokens.len(),
                        p.chunk_offset,
                        p.prompt_tokens.len(),
                    );
                    prefilling.push(p);
                }
                // EOS on first token: the whole prompt ran.
                Ok(StartPrefillResult::Finished) => ledger.record_chunk(prompt_len, 0),
                Err(e) => {
                    ledger.taint();
                    handle_prefill_start_error(&e, active.len());
                }
            }
        } else {
            ledger.taint();
            // Legacy non-chunked path.
            match prefill_request(
                think_end_token,
                think_start_token,
                code_fence_token,
                tool_call_start_token,
                tool_call_end_token,
                model,
                req,
                eos_tokens,
                grammar_engine,
                spontaneous_think_budget,
            ) {
                Ok(Some(a)) => {
                    tracing::info!(
                        "Prefilled: seq_len={}, remaining={}",
                        a.seq.seq_len,
                        a.remaining,
                    );
                    active.push(a);
                }
                Ok(None) => {}
                Err(e) => {
                    handle_prefill_start_error(&e, active.len());
                }
            }
        }
    }
    if defer_first_stream_event && active.len() > active_start {
        let cohort_start = Instant::now();
        let mut idx = active_start;
        while idx < active.len() {
            let a = &mut active[idx];
            a.decode_start = cohort_start;
            a.last_token_time = cohort_start;
            let deferred = !a.output_tokens.is_empty();
            let send_ok = if deferred && let ResponseSink::Streaming(ref tx) = a.sink {
                let first = a.last_token;
                let first_lp = a.logprobs_data.first().cloned();
                let is_eos = a.eos_tokens.contains(&first);
                stream_send_bounded(
                    tx,
                    first_token_event(first, &first_lp, is_eos),
                    "prefill_a_step cohort first token",
                )
            } else {
                true
            };
            if !send_ok {
                let mut rejected = active.remove(idx);
                if let Err(error) = rejected
                    .continuation_record_mut()
                    .finish_or_release(model, false)
                {
                    tracing::error!(
                        "phase_start_prefills: free_sequence after cohort first-token rejection: {error:#}"
                    );
                }
                if let Err(error) = model.ep_broadcast_cmd(0xFFFFFFF1) {
                    tracing::error!(
                        "phase_start_prefills: EP reset after cohort first-token rejection: {error:#}"
                    );
                }
            } else {
                idx += 1;
            }
        }
    }
    crate::metrics::compat_metrics::set_scheduler_counts(
        active.len() + prefilling.len(),
        pending.0.lock().requests.len(),
    );
    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());
}

/// Log a prefill start failure. The failing request's sink has ALREADY been
/// answered inside `start_chunked_prefill` / `prefill_request` (every error
/// path there calls `send_error_to_sink`), so nothing else is owed to it.
///
/// Previously an "SSM pool exhausted" error here ALSO killed the newest
/// active sequence ("Preempted: server resource pressure"). That preemption
/// helped nobody: the new request had already been rejected, so the victim's
/// partial generation was thrown away for a slot nothing was waiting to use.
/// Admission caps `active + prefilling` at `max_batch_size`, which is what the
/// SSM pool is sized for, so pool exhaustion here is the symptom of a slot
/// leak (see the retire-path compaction fix), and the victim kill turned one
/// leaked slot into a cascade of killed healthy requests.
fn handle_prefill_start_error(e: &anyhow::Error, active_len: usize) {
    let err_msg = format!("{e:#}");
    if err_msg.contains("pool exhausted") {
        tracing::error!(
            active = active_len,
            "Prefill start error: {err_msg} (request rejected; active sequences left untouched — \
             a pool exhausted below max_batch_size indicates a leaked SSM slot)"
        );
    } else {
        tracing::error!("Prefill start error: {err_msg}");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;
    use crate::scheduler::test_fixtures::{active_seq, blocking_request, model};

    fn empty_pending() -> std::sync::Arc<(Mutex<PendingQueue>, Condvar)> {
        std::sync::Arc::new((
            Mutex::new(PendingQueue {
                requests: Vec::new(),
                closed: false,
            }),
            Condvar::new(),
        ))
    }

    fn streaming_request(
        prompt_tokens: Vec<u32>,
    ) -> (InferenceRequest, tokio::sync::mpsc::Receiver<StreamEvent>) {
        let (token_tx, rx) = tokio::sync::mpsc::channel(16);
        let req = InferenceRequest::Streaming {
            prompt_tokens,
            session_hash: 0,
            image_pixels: Vec::new(),
            max_tokens: 8,
            min_tokens: 0,
            ignore_eos: false,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            top_n_sigma: 0.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 3,
            lz_penalty: 0.0,
            logit_bias: Vec::new(),
            stop_tokens: Vec::new(),
            enable_thinking: false,
            thinking_budget: None,
            require_tool_call: false,
            suppress_tool_call: false,
            disable_mtp: false,
            grammar_spec: None,
            seed: None,
            top_logprobs: None,
            timeout_at: None,
            decode_timeout_at: None,
            intervention_ledger: None,
            think_nudge: None,
            token_tx,
            cancel_flag: std::sync::Arc::new(AtomicBool::new(false)),
        };
        (req, rx)
    }

    /// Minimal Qwen4-shaped mock `Model` for exercising `prefill_batch_chunk`
    /// bursts, distinct logits rows, error drains, and slot-leak invariants.
    struct Qwen4ShortBatchMock {
        next_slot: AtomicUsize,
        alloc_calls: AtomicUsize,
        fail_alloc_on: std::sync::Mutex<Option<usize>>,
        batch_error: std::sync::Mutex<Option<String>>,
        batch_calls: std::sync::Mutex<Vec<Vec<usize>>>,
        serial_calls: std::sync::Mutex<Vec<usize>>,
        frees: std::sync::Mutex<Vec<usize>>,
        events: std::sync::Mutex<Vec<String>>,
        total_blocks: usize,
        free_blocks: usize,
    }

    impl Qwen4ShortBatchMock {
        fn new(total_blocks: usize, free_blocks: usize) -> Self {
            Self {
                next_slot: AtomicUsize::new(0),
                alloc_calls: AtomicUsize::new(0),
                fail_alloc_on: std::sync::Mutex::new(None),
                batch_error: std::sync::Mutex::new(None),
                batch_calls: std::sync::Mutex::new(Vec::new()),
                serial_calls: std::sync::Mutex::new(Vec::new()),
                frees: std::sync::Mutex::new(Vec::new()),
                events: std::sync::Mutex::new(Vec::new()),
                total_blocks,
                free_blocks,
            }
        }

        fn token_for_logits_ptr(ptr: DevicePtr) -> u32 {
            if ptr.0 >= 0x1000 {
                ((ptr.0 - 0x1000) / 0x100) as u32 + 100
            } else {
                42
            }
        }
    }

    impl Model for Qwen4ShortBatchMock {
        fn hc_mult(&self) -> usize {
            4
        }

        fn is_ep(&self) -> bool {
            false
        }

        fn qwen4_batch_plan(
            &self,
        ) -> Result<Option<spark_model::qwen4_batch_policy::Qwen4BatchPlan>> {
            Ok(Some(spark_model::qwen4_batch_policy::Qwen4BatchPlan {
                selector: spark_model::qwen4_batch_policy::NativeBatchSelector::Auto,
                process_reason: None,
                qsa_rows: false,
            }))
        }

        fn num_total_blocks(&self) -> usize {
            self.total_blocks
        }

        fn num_free_blocks(&self) -> usize {
            self.free_blocks
        }

        fn prefill(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
            unreachable!()
        }

        fn prefill_chunk(
            &self,
            tokens: &[u32],
            seq: &mut SequenceState,
            chunk_start: usize,
            chunk_len: usize,
            _: bool,
            _: u64,
        ) -> Result<DevicePtr> {
            self.serial_calls.lock().unwrap().push(chunk_len);
            seq.tokens
                .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
            seq.seq_len = chunk_start + chunk_len;
            Ok(DevicePtr(0x1000 + (seq.slot_idx as u64) * 0x100))
        }

        fn prefill_batch_chunk(
            &self,
            streams: &mut [PrefillSlice<'_>],
            _: u64,
        ) -> Result<Vec<DevicePtr>> {
            let lens: Vec<usize> = streams.iter().map(|s| s.chunk_len).collect();
            self.batch_calls.lock().unwrap().push(lens);
            if let Some(msg) = self.batch_error.lock().unwrap().clone() {
                anyhow::bail!("{msg}");
            }
            let mut out = Vec::with_capacity(streams.len());
            for (b, slice) in streams.iter_mut().enumerate() {
                slice.seq.tokens.extend_from_slice(
                    &slice.prompt_tokens[slice.chunk_start..slice.chunk_start + slice.chunk_len],
                );
                slice.seq.seq_len = slice.chunk_start + slice.chunk_len;
                out.push(DevicePtr(0x1000 + (b as u64) * 0x100));
            }
            Ok(out)
        }

        fn decode(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
            unreachable!()
        }

        fn decode_batch(
            &self,
            _: &[u32],
            _: &mut [&mut SequenceState],
            _: u64,
        ) -> Result<DevicePtr> {
            unreachable!()
        }

        fn vocab_size(&self) -> usize {
            256
        }

        fn bind_gpu_to_thread(&self) -> Result<()> {
            Ok(())
        }

        fn alloc_sequence(&self) -> Result<SequenceState> {
            let idx = self.alloc_calls.fetch_add(1, Ordering::SeqCst);
            if *self.fail_alloc_on.lock().unwrap() == Some(idx) {
                anyhow::bail!("injected alloc failure at index {idx}");
            }
            let slot = self.next_slot.fetch_add(1, Ordering::SeqCst);
            let mut s = model::seq_state();
            s.slot_idx = slot;
            s.seq_len = 0;
            Ok(s)
        }

        fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
            dst.fill(0);
            let win = Self::token_for_logits_ptr(logits_ptr) as usize;
            let bits = (10.0f32.to_bits() >> 16) as u16;
            dst[win * 2..win * 2 + 2].copy_from_slice(&bits.to_le_bytes());
            Ok(())
        }

        fn logits_buffer_ptr(&self) -> DevicePtr {
            DevicePtr(0x1000)
        }

        fn argmax_on_device(&self, logits_ptr: DevicePtr, _: u64) -> Result<u32> {
            Ok(Self::token_for_logits_ptr(logits_ptr))
        }

        fn argmax_batch(&self, _: DevicePtr, _: usize, _: u64) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn hidden_after_norm(&self) -> DevicePtr {
            DevicePtr::NULL
        }

        fn decode_verify(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn checkpoint_ssm_states(&self, _: &mut SequenceState) -> Result<()> {
            unreachable!()
        }

        fn rollback_ssm_states(&self, _: &mut SequenceState, _: usize) -> Result<()> {
            unreachable!()
        }

        fn generate_speculative(
            &self,
            _: &[u32],
            _: &spark_runtime::sampler::SamplingParams,
            _: usize,
        ) -> Result<spark_model::engine::GenerateResult> {
            unreachable!()
        }

        fn has_proposer(&self) -> bool {
            false
        }

        fn has_self_speculative(&self) -> bool {
            false
        }

        fn decode_draft(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
            unreachable!()
        }

        fn cache_sequence(&self, _: &SequenceState) {}

        fn free_sequence(&self, seq: &mut SequenceState) -> Result<()> {
            self.frees.lock().unwrap().push(seq.slot_idx);
            self.events
                .lock()
                .unwrap()
                .push(format!("free:{}", seq.slot_idx));
            Ok(())
        }

        fn compact_sequence(&self, _: &mut SequenceState, _: usize) -> Result<bool> {
            Ok(true)
        }

        fn decode_verify_graphed(
            &self,
            _: &[u32; 2],
            _: &mut SequenceState,
            _: u64,
        ) -> Result<[u32; 2]> {
            unreachable!()
        }

        fn decode_verify_graphed_k3(
            &self,
            _: &[u32; 3],
            _: &mut SequenceState,
            _: u64,
        ) -> Result<[u32; 3]> {
            unreachable!()
        }

        fn decode_verify_graphed_k4(
            &self,
            _: &[u32; 4],
            _: &mut SequenceState,
            _: u64,
        ) -> Result<[u32; 4]> {
            unreachable!()
        }

        fn save_hidden_for_mtp(&self, _: usize, _: u64) -> Result<()> {
            unreachable!()
        }

        fn run_mtp_propose(
            &self,
            _: u32,
            _: usize,
            _: &mut SequenceState,
            _: u64,
        ) -> Result<Option<u32>> {
            unreachable!()
        }

        fn run_mtp_propose_multi(
            &self,
            _: u32,
            _: usize,
            _: usize,
            _: &mut SequenceState,
            _: u64,
            _: Option<&[i32]>,
        ) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn trim_proposer_state(&self, _: &mut SequenceState, _: usize, _: u64) -> Result<()> {
            unreachable!()
        }

        fn synchronize_stream(&self, stream: u64) -> Result<()> {
            self.events.lock().unwrap().push(format!("sync:{stream}"));
            Ok(())
        }
    }

    #[test]
    fn initial_prefill_never_consumes_decode_slot_capacity() {
        assert_eq!(initial_prefill_chunk_budget(8_192, 8_193), 8_192);
        assert_eq!(initial_prefill_chunk_budget(2_048, 8_193), 2_048);
    }

    #[test]
    fn short_batch_kill_switch_defaults_on_and_disables_only_on_zero() {
        assert!(parse_short_batch_enabled(None));
        assert!(parse_short_batch_enabled(Some("1")));
        assert!(parse_short_batch_enabled(Some("")));
        assert!(!parse_short_batch_enabled(Some("0")));
        assert!(!parse_short_batch_enabled(Some(" 0 ")));
    }

    #[test]
    fn short_batch_prefix_selector_enforces_count_rows_len_kv_vision_and_fifo() {
        let cand = |len: usize| ShortBatchCandidate {
            prompt_len: len,
            single_chunk_budget: 2048,
            has_images: false,
            has_disagg: false,
        };
        let limits = ShortBatchLimits {
            max_streams: 32,
            max_total_rows: 4096,
            block_size: 16,
            avail_kv_blocks: Some(1024),
        };

        // Fewer than 2 streams -> 0.
        assert_eq!(select_short_batch_prefix(&[cand(64)], limits), 0);

        // Caps at 16 streams even when 20 short requests are queued.
        let twenty = vec![cand(64); 20];
        assert_eq!(select_short_batch_prefix(&twenty, limits), 16);

        // Enforces 2048 total-row hard cap even if max_total_rows is higher.
        let five_512 = vec![cand(512); 5];
        assert_eq!(select_short_batch_prefix(&five_512, limits), 4);

        // chunk_len < 2 stops FIFO collection immediately.
        let single_tok_front = vec![cand(1), cand(64), cand(64)];
        assert_eq!(select_short_batch_prefix(&single_tok_front, limits), 0);

        // Vision or disagg at index 2 stops after collecting [0, 1] without reordering.
        let mut vision_mid = vec![cand(64); 4];
        vision_mid[2].has_images = true;
        assert_eq!(select_short_batch_prefix(&vision_mid, limits), 2);

        let mut disagg_front = vec![cand(64); 4];
        disagg_front[0].has_disagg = true;
        assert_eq!(select_short_batch_prefix(&disagg_front, limits), 0);

        // Multi-chunk request (prompt_len > single_chunk_budget) stops collection.
        let mut multi_chunk = vec![cand(64); 4];
        multi_chunk[2].single_chunk_budget = 32;
        assert_eq!(select_short_batch_prefix(&multi_chunk, limits), 2);

        // KV block budget bounds the collected prefix.
        let tight_kv = ShortBatchLimits {
            avail_kv_blocks: Some(12), // each 64-tok prompt needs 4 blocks -> fits 3
            ..limits
        };
        assert_eq!(select_short_batch_prefix(&twenty, tight_kv), 3);

        // Available batch slots bound the collected prefix.
        let tight_slots = ShortBatchLimits {
            max_streams: 3,
            ..limits
        };
        assert_eq!(select_short_batch_prefix(&twenty, tight_slots), 3);
    }

    #[test]
    fn short_batch_dispatches_c5_burst_via_prefill_batch_chunk_with_distinct_tokens() {
        let m = Qwen4ShortBatchMock::new(512, 512);
        let pending = empty_pending();
        let mut active = Vec::new();
        let mut prefilling = Vec::new();
        let mut grammar_engine = None;
        let mut ledger = super::super::prefill_slo::PrefillTickLedger::default();

        let lens = [32usize, 48, 64, 24, 40];
        let mut reqs = Vec::new();
        let mut rxs = Vec::new();
        for len in lens {
            let (req, rx) = streaming_request(vec![7; len]);
            reqs.push(req);
            rxs.push(rx);
        }

        start_new_requests_with_short_batch(
            &m,
            reqs,
            &pending,
            &[],
            usize::MAX,
            0,
            16,
            0,
            16,
            true,
            2048,
            4096,
            super::super::prefill_chunk_policy::ChunkPlan::default(),
            &[17],
            3,
            9,
            &mut grammar_engine,
            512,
            None,
            None,
            None,
            None,
            None,
            &mut active,
            &mut prefilling,
            &mut ledger,
            true,
        );

        assert_eq!(*m.batch_calls.lock().unwrap(), vec![lens.to_vec()]);
        assert!(m.serial_calls.lock().unwrap().is_empty());
        assert!(m.frees.lock().unwrap().is_empty());
        assert!(prefilling.is_empty());
        assert_eq!(active.len(), 5);
        for (idx, (seq, rx)) in active.iter().zip(rxs.iter_mut()).enumerate() {
            let expected_tok = 100 + idx as u32;
            assert_eq!(seq.last_token, expected_tok);
            assert_eq!(seq.seq.seq_len, lens[idx]);
            match rx.try_recv() {
                Ok(StreamEvent::Token(tok)) => assert_eq!(tok, expected_tok),
                Ok(_) => panic!("expected first StreamEvent::Token for stream {idx}"),
                Err(err) => panic!("missing first token event for stream {idx}: {err}"),
            }
        }
    }

    #[test]
    fn short_batch_kill_switch_zero_routes_to_serial_prefill_chunk() {
        let m = Qwen4ShortBatchMock::new(512, 512);
        let pending = empty_pending();
        let mut active = Vec::new();
        let mut prefilling = Vec::new();
        let mut grammar_engine = None;
        let mut ledger = super::super::prefill_slo::PrefillTickLedger::default();

        let mut reqs = Vec::new();
        let mut _rxs = Vec::new();
        for len in [32usize, 40, 48, 56] {
            let (req, rx) = blocking_request(vec![5; len]);
            reqs.push(req);
            _rxs.push(rx);
        }

        start_new_requests_with_short_batch(
            &m,
            reqs,
            &pending,
            &[],
            usize::MAX,
            0,
            16,
            0,
            16,
            true,
            2048,
            4096,
            super::super::prefill_chunk_policy::ChunkPlan::default(),
            &[17],
            3,
            9,
            &mut grammar_engine,
            512,
            None,
            None,
            None,
            None,
            None,
            &mut active,
            &mut prefilling,
            &mut ledger,
            false,
        );

        assert!(m.batch_calls.lock().unwrap().is_empty());
        assert_eq!(*m.serial_calls.lock().unwrap(), vec![32, 40, 48, 56]);
        assert_eq!(active.len(), 4);
        assert!(m.frees.lock().unwrap().is_empty());
    }

    #[test]
    fn short_batch_error_drains_answers_all_sinks_and_frees_every_claimed_slot() {
        let m = Qwen4ShortBatchMock::new(512, 512);
        *m.batch_error.lock().unwrap() = Some("injected short-batch CUDA failure".into());
        let pending = empty_pending();
        let mut active = Vec::new();
        let mut prefilling = Vec::new();
        let mut grammar_engine = None;
        let mut ledger = super::super::prefill_slo::PrefillTickLedger::default();

        let mut reqs = Vec::new();
        let mut rxs = Vec::new();
        for _ in 0..3 {
            let (req, rx) = blocking_request(vec![9; 32]);
            reqs.push(req);
            rxs.push(rx);
        }

        start_new_requests_with_short_batch(
            &m,
            reqs,
            &pending,
            &[],
            usize::MAX,
            0,
            16,
            0,
            16,
            true,
            2048,
            4096,
            super::super::prefill_chunk_policy::ChunkPlan::default(),
            &[17],
            7,
            9,
            &mut grammar_engine,
            512,
            None,
            None,
            None,
            None,
            None,
            &mut active,
            &mut prefilling,
            &mut ledger,
            true,
        );

        assert!(active.is_empty());
        assert!(prefilling.is_empty());
        assert_eq!(*m.frees.lock().unwrap(), vec![0, 1, 2]);
        let events = m.events.lock().unwrap().clone();
        assert_eq!(
            events,
            vec!["sync:7", "sync:0", "free:0", "free:1", "free:2"],
            "stream barrier must drain before freeing any claimed short-batch slot"
        );
        for (idx, rx) in rxs.iter_mut().enumerate() {
            match rx.try_recv() {
                Ok(Err(err)) => {
                    assert!(
                        err.to_string()
                            .contains("injected short-batch CUDA failure"),
                        "stream {idx} unexpected error: {err:#}"
                    );
                }
                Ok(Ok(_)) => panic!("stream {idx} must receive error"),
                Err(_) => panic!("stream {idx} must be answered, not dropped"),
            }
        }
    }

    #[test]
    fn short_batch_mid_claim_alloc_failure_restores_tail_without_leaking_slots() {
        let m = Qwen4ShortBatchMock::new(512, 512);
        *m.fail_alloc_on.lock().unwrap() = Some(2);
        let pending = empty_pending();
        let mut active = Vec::new();
        let mut prefilling = Vec::new();
        let mut grammar_engine = None;
        let mut ledger = super::super::prefill_slo::PrefillTickLedger::default();

        let mut reqs = Vec::new();
        let mut rxs = Vec::new();
        for len in [20usize, 24, 28, 32] {
            let (req, rx) = blocking_request(vec![3; len]);
            reqs.push(req);
            rxs.push(rx);
        }

        start_new_requests_with_short_batch(
            &m,
            reqs,
            &pending,
            &[],
            usize::MAX,
            0,
            16,
            0,
            16,
            true,
            2048,
            4096,
            super::super::prefill_chunk_policy::ChunkPlan::default(),
            &[17],
            3,
            9,
            &mut grammar_engine,
            512,
            None,
            None,
            None,
            None,
            None,
            &mut active,
            &mut prefilling,
            &mut ledger,
            true,
        );

        // First two claimed requests ran via prefill_batch_chunk; request 2 got
        // its alloc error; request 3 was restored to FIFO front and ran serially.
        assert_eq!(*m.batch_calls.lock().unwrap(), vec![vec![20, 24]]);
        assert_eq!(*m.serial_calls.lock().unwrap(), vec![32]);
        assert!(m.frees.lock().unwrap().is_empty());
        assert_eq!(active.len(), 3);
        assert_eq!(active[0].seq.seq_len, 20);
        assert_eq!(active[1].seq.seq_len, 24);
        assert_eq!(active[2].seq.seq_len, 32);
        match rxs[2].try_recv() {
            Ok(Err(err)) => {
                assert!(
                    err.to_string().contains("injected alloc failure"),
                    "{err:#}"
                );
            }
            Ok(Ok(_)) => panic!("request 2 must receive alloc error"),
            Err(_) => panic!("request 2 must be answered, not dropped"),
        }
    }

    /// H1: when the pool cannot allocate a slot for a new request, that
    /// request is answered with the allocation error and nothing else
    /// happens. Previously the newest active sequence was also killed
    /// ("Preempted: server resource pressure") although the rejected
    /// request was already gone and nothing could use the freed slot.
    #[test]
    fn pool_exhaustion_rejects_the_new_request_and_leaves_live_sequences_alone() {
        let m = model::recording();
        *m.alloc_error.lock().unwrap() = Some("SSM state pool exhausted".into());
        let (victim_tx, mut victim_rx) = tokio::sync::oneshot::channel();
        let mut active = vec![active_seq(ResponseSink::Blocking(Some(victim_tx)), 0)];
        let mut prefilling = Vec::new();
        let (req, mut rx) = blocking_request(vec![1, 2, 3]);
        let mut grammar_engine = None;
        let pending = empty_pending();

        start_new_requests(
            &m,
            vec![req],
            &pending,
            &[],
            usize::MAX,
            0,
            16,
            0,
            8,
            false,
            512,
            512,
            super::super::prefill_chunk_policy::ChunkPlan::default(),
            &[17],
            0,
            0,
            &mut grammar_engine,
            512,
            None,
            None,
            None,
            None,
            None,
            &mut active,
            &mut prefilling,
            &mut super::super::prefill_slo::PrefillTickLedger::default(),
        );

        let err = match rx.try_recv() {
            Ok(Err(err)) => err,
            Ok(Ok(_)) => panic!("an allocation failure is an error"),
            Err(_) => panic!("the rejected request is answered, not dropped"),
        };
        assert!(err.to_string().contains("pool exhausted"), "{err:#}");
        assert_eq!(active.len(), 1, "no healthy sequence is preempted");
        assert!(
            matches!(
                victim_rx.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            ),
            "the live sequence's client hears nothing"
        );
        assert!(m.frees.lock().unwrap().is_empty(), "nothing is freed");
        assert!(prefilling.is_empty());
    }
}
