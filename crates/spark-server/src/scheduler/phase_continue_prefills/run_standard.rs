// SPDX-License-Identifier: AGPL-3.0-only

//! Single-stream chunked-prefill body.
//!
//! Tries `mixed_forward` first when an active decode lane is available
//! (fuses 1 prefill chunk + N decode in one pass). Otherwise falls back
//! to plain `prefill_chunk` + EP broadcast.

use anyhow::Result;
use spark_model::traits::{Model, SequenceState};
use std::time::Instant;

use super::super::decode_logits_step::process_decode_logits;
use super::super::prefill_a_step::prefill_first_token_applies_grammar;
use super::super::prefill_chunk_policy;
use super::super::prefill_slo::PrefillTickLedger;
use super::super::prefill_stream_barrier::{
    abort_after_mixed_forward_error, complete_prefill_handoff, drain_before_prefill_cleanup,
};
use super::super::sample_token_with_logprobs_and_grammar;
use super::super::types::{ActiveSeq, PrefillInProgress};

fn no_mix_bisect_enabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("ATLAS_BISECT_NO_MIX")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_standard_chunk_loop(
    model: &dyn Model,
    p: &mut PrefillInProgress,
    idx: usize,
    active: &mut Vec<ActiveSeq>,
    max_prefill_tokens: usize,
    chunk_plan: prefill_chunk_policy::ChunkPlan,
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
    completed_indices: &mut Vec<(usize, Option<u32>)>,
    did_mixed_step: &mut bool,
    ledger: &mut PrefillTickLedger,
) {
    // TQ+ InnerQ: poll once per chunk to see if calibration has banked
    // enough K² stats to finalize scales. The driver itself is idempotent
    // (no-op after activation), so calling on every chunk is cheap.
    // Group size = 128 = Qwen3 head_dim (the only currently-supported shape).
    super::poll_innerq();
    // Single chunk per call — the outer scheduler loop re-enters this
    // function on the very next iteration to advance the next stream
    // or the next chunk. This yield keeps fairness across pending
    // requests (Q12: avoids back-to-back chunked prefill monopolising
    // the scheduler).
    let remaining = p.prompt_tokens.len() - p.chunk_offset;
    // Chunk sizing lives in `prefill_chunk_policy`: the full chunk while the
    // box is idle, the bounded interleave width (a prepared fast-prefill
    // width aligned to the offset) or the active-decode cap while any row
    // decodes (bounds the per-tick decode stall to one bounded chunk), WY4
    // alignment for intermediate chunks, and the MLA single-chunk rule. MLA: Flux has no
    // `prefill_attention_paged_mla_*` kernel; the existing MLA prefill in
    // `qwen3_attention::prefill` only attends over the current chunk's
    // K/V, so multi-chunk prefill silently corrupts attention output. Force
    // single-chunk until a paged-MLA prefill kernel lands.
    let req_max_prefill_tokens = if chunk_plan.widths.is_none() && p.chunk_budget > 0 {
        let stepped = max_prefill_tokens.min(p.chunk_budget);
        p.chunk_budget = stepped;
        stepped
    } else {
        max_prefill_tokens
    };
    let chunk_len = prefill_chunk_policy::prefill_chunk_len(prefill_chunk_policy::ChunkRequest {
        offset: p.chunk_offset,
        remaining,
        max_prefill_tokens: req_max_prefill_tokens,
        plan: chunk_plan,
        decode_rows_active: !active.is_empty(),
        single_chunk_required: model.is_mla(),
    });
    let is_last = p.chunk_offset + chunk_len >= p.prompt_tokens.len();

    // ── Mixed forward: fuse prefill chunk + decode in one pass ──
    // ATLAS_BISECT_NO_MIX=1 forces this branch to false so we can
    // diagnose whether the chunked-prefill+concurrent CUDA-700 lives
    // inside `mixed_forward` (active+prefill fused) vs the pure
    // decode-batch path.
    let no_mix_bisect = no_mix_bisect_enabled();
    let can_mix = !no_mix_bisect
        && !active.is_empty()
        && !model.is_ep()
        && !use_mtp
        && !use_self_speculative
        && !use_ngram_speculative
        // `mixed_forward` enters the generic multi-row layer paths, which do
        // not yet implement Qwen4/mHC PLE, highway, or active-QSA semantics.
        && model.hc_mult() == 0;

    if can_mix {
        // Fused prefill+decode: its wall time is not "one chunk, then decode".
        ledger.taint();
        let decode_tokens: Vec<u32> = active.iter().map(|a| a.last_token).collect();
        let mut decode_refs: Vec<&mut SequenceState> =
            active.iter_mut().map(|a| &mut a.seq).collect();
        let t0_mixed = Instant::now();

        match model.mixed_forward(
            &decode_tokens,
            &mut decode_refs,
            &p.prompt_tokens,
            &mut p.seq,
            p.chunk_offset,
            chunk_len,
            is_last,
            prefill_stream,
        ) {
            Ok(result) => {
                p.chunk_offset += chunk_len;
                crate::metrics::record_prefill_progress(
                    p.session_hash,
                    p.chunk_offset,
                    p.prompt_tokens.len(),
                    chunk_len,
                );
                tracing::info!(
                    "Mixed forward: prefill {}/{} tokens + {} decode",
                    p.chunk_offset,
                    p.prompt_tokens.len(),
                    decode_tokens.len(),
                );

                // Process prefill logits (if last chunk).
                if is_last {
                    if let Err(e) = model.normalize_ssm_states(&p.seq, prefill_stream) {
                        tracing::warn!("SSM state normalization failed: {e:#}");
                    }
                    complete_prefill_handoff(
                        model,
                        prefill_stream,
                        prefill_event,
                        "mixed prefill sampling handoff",
                    );
                    let params = p.sampling_params();
                    let apply_grammar =
                        prefill_first_token_applies_grammar(p.enable_thinking, think_end_token);
                    match sample_token_with_logprobs_and_grammar(
                        model,
                        result.prefill_logits,
                        &params,
                        &p.eos_tokens,
                        &[],
                        p.top_logprobs,
                        p.grammar_state.as_mut(),
                        apply_grammar,
                    ) {
                        Ok((first, first_lp)) => {
                            super::super::logits_row_receipt::prefill_raw(
                                model,
                                result.prefill_logits,
                                p.session_hash,
                                first,
                            );
                            p.first_logprobs = first_lp;
                            tracing::info!("Mixed prefill first token: {first}");
                            completed_indices.push((idx, Some(first)));
                        }
                        Err(e) => {
                            drain_before_prefill_cleanup(
                                model,
                                prefill_stream,
                                "mixed prefill sampling failure",
                            );
                            tracing::error!("Mixed prefill sampling: {e:#}");
                            completed_indices.push((idx, None));
                        }
                    }
                }

                // Process decode logits for active sequences.
                complete_prefill_handoff(
                    model,
                    prefill_stream,
                    prefill_event,
                    "mixed decode handoff",
                );
                process_decode_logits(
                    model,
                    active,
                    result.decode_logits,
                    t0_mixed,
                    think_end_token,
                    think_start_token,
                    code_fence_token,
                    tool_call_start_token,
                    tool_call_end_token,
                    reflection_suppress_ids,
                    adaptive_sampling,
                    crate::scheduler::helpers::honor_eos_inside_thinking(),
                );
                *did_mixed_step = true;
            }
            Err(e) => {
                abort_after_mixed_forward_error(model, prefill_stream, "mixed forward", &e);
            }
        }
        return;
    }

    // ── Standard path: prefill chunk only, decode separately ──
    // EP: broadcast chunk tokens to worker (bulk, single NCCL op).
    let ep_ok = (|| -> Result<()> {
        model.ep_broadcast_cmd(0xFFFFFFF0)?;
        model.ep_broadcast_cmd(chunk_len as u32)?;
        model.ep_broadcast_cmd(p.chunk_offset as u32)?;
        model.ep_broadcast_cmd(p.prompt_tokens.len() as u32)?;
        model.ep_broadcast_tokens(&p.prompt_tokens)?;
        Ok(())
    })();
    if let Err(e) = ep_ok {
        drain_before_prefill_cleanup(model, prefill_stream, "prefill EP broadcast failure");
        tracing::error!("EP broadcast chunk: {e:#}");
        completed_indices.push((idx, None));
        return;
    }

    match model.prefill_chunk(
        &p.prompt_tokens,
        &mut p.seq,
        p.chunk_offset,
        chunk_len,
        is_last,
        prefill_stream,
    ) {
        Ok(logits) => {
            ledger.record_chunk(chunk_len, p.chunk_offset);
            p.chunk_offset += chunk_len;
            crate::metrics::record_prefill_progress(
                p.session_hash,
                p.chunk_offset,
                p.prompt_tokens.len(),
                chunk_len,
            );
            tracing::info!(
                "Prefill chunk {}/{} tokens",
                p.chunk_offset,
                p.prompt_tokens.len(),
            );
            // Normalize SSM states after EVERY chunk to prevent state drift.
            if let Err(e) = model.normalize_ssm_states(&p.seq, prefill_stream) {
                tracing::warn!("SSM state normalization failed: {e:#}");
            }
            if is_last {
                complete_prefill_handoff(
                    model,
                    prefill_stream,
                    prefill_event,
                    "chunked prefill handoff",
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
                        super::super::logits_row_receipt::prefill_raw(
                            model,
                            logits,
                            p.session_hash,
                            first,
                        );
                        p.first_logprobs = first_lp;
                        tracing::info!("Prefill first token: {first}");
                        completed_indices.push((idx, Some(first)));
                    }
                    Err(e) => {
                        drain_before_prefill_cleanup(
                            model,
                            prefill_stream,
                            "chunked prefill sampling failure",
                        );
                        tracing::error!("Chunked prefill sample_token: {e:#}");
                        completed_indices.push((idx, None));
                    }
                }
            }
        }
        Err(e) => {
            ledger.taint();
            drain_before_prefill_cleanup(model, prefill_stream, "chunked prefill failure");
            tracing::error!("Prefill chunk error: {e:#}");
            completed_indices.push((idx, None));
        }
    }
}
