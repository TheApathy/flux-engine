// SPDX-License-Identifier: AGPL-3.0-only

//! MTP speculative draft proposal step.

use super::*;
use proposal_lifecycle::{clear_orphan_aux, install_collected, propose_and_install, retain_prefix};

#[path = "mtp_bootstrap_failure.rs"]
mod bootstrap_failure;
use bootstrap_failure::fail_bootstrap;
pub(super) use bootstrap_failure::fail_speculative_step;

#[path = "history_ring.rs"]
pub(crate) mod history_ring;

#[cfg(test)]
#[path = "mtp_verifier_failure_tests.rs"]
mod verifier_failure_tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerifyDispatch {
    Bootstrap,
    K2,
    K3,
    K4,
    Generic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MtpWork {
    Bootstrap,
    Verify,
}

/// Finished sequences remain in `active` until the scheduler's retirement
/// phase. They must never re-enter either MTP path during that interval: a
/// bootstrap would sample past EOS/ChatML and a verify would consume work
/// after the response had already terminated.
fn mtp_work(finished: bool, has_pending_drafts: bool) -> Option<MtpWork> {
    if finished {
        None
    } else if has_pending_drafts {
        Some(MtpWork::Verify)
    } else {
        Some(MtpWork::Bootstrap)
    }
}

/// Pick the verifier from draft width and whether raw argmax is policy-safe.
fn select_verify_dispatch(
    draft_len: usize,
    policy_required: bool,
    has_tree: bool,
    has_sampled: bool,
) -> VerifyDispatch {
    if draft_len >= 4 {
        return VerifyDispatch::Generic;
    }
    if draft_len == 0 {
        return VerifyDispatch::Bootstrap;
    }
    if policy_required || has_tree || has_sampled {
        return VerifyDispatch::Generic;
    }
    match draft_len {
        1 => VerifyDispatch::K2,
        2 => VerifyDispatch::K3,
        3 => VerifyDispatch::K4,
        _ => VerifyDispatch::Bootstrap,
    }
}

pub(super) fn verify_policy_required(a: &ActiveSeq, think: &ThinkSpecCtx<'_>) -> bool {
    if a.inside_thinking {
        think.enabled
    } else {
        content_policy_required(a, think)
    }
}

fn batched_dflash_verify_allowed(
    enabled: bool,
    sequence_count: usize,
    is_ep: bool,
    proposer_is_dflash: bool,
) -> bool {
    enabled && sequence_count >= 2 && !is_ep && proposer_is_dflash
}

/// A probabilistic DFlash request can finish on the bootstrap token before
/// there is any legal opportunity to call the proposer.  Distinguish that
/// truthful structural stop from every other early-finish path: the sampled
/// token must have been committed exactly once and it must itself be EOS or
/// the ChatML role boundary.  In particular, cancellation, decode/D2H errors,
/// timeouts, watchdog stops, and max-token exhaustion are not benign
/// zero-proposal outcomes.
fn sampled_bootstrap_plain_structural_boundary(
    probabilistic_dflash: bool,
    finished: bool,
    output_len_before_bootstrap: usize,
    output_tokens: &[u32],
    tok: u32,
    eos_tokens: &[u32],
    im_start: Option<u32>,
) -> bool {
    probabilistic_dflash
        && finished
        && output_tokens.len() == output_len_before_bootstrap.saturating_add(1)
        && output_tokens.last().copied() == Some(tok)
        && (eos_tokens.contains(&tok) || im_start == Some(tok))
}

fn notify_qwen4_external_draft(a: &mut ActiveSeq, chain: &[u32]) {
    a.seq.retire_committed_verify_handoff();
    if let Some(prop_state) = a.seq.proposer_state.as_mut()
        && let Some(mtp_state) = prop_state
            .as_any_mut()
            .downcast_mut::<spark_model::layers::qwen4_mtp::Qwen4MtpState>()
    {
        mtp_state.notify_external_draft_with_tokens(chain);
    }
}

fn notify_qwen4_grafted_suffix(a: &mut ActiveSeq, suffix: &[u32]) {
    if let Some(prop_state) = a.seq.proposer_state.as_mut()
        && let Some(mtp_state) = prop_state
            .as_any_mut()
            .downcast_mut::<spark_model::layers::qwen4_mtp::Qwen4MtpState>()
    {
        mtp_state.notify_grafted_suffix(suffix);
    }
}

/// Retire pending DFlash target-hidden captures into the sequence's context
/// ring before bypassing `DraftProposer::propose` (e.g., external retrieval,
/// history-ring copy/fallback, forced-close nudge, or grammar-gated skip).
///
/// Without retiring those rows before another decode or verify overwrites
/// `dflash_hidden_save`, `dstate.ctx_len` falls behind `a.seq.seq_len` and the
/// next proposal/bootstrap fails closed with a cursor-debt mismatch. Returns
/// `true` when it is safe to proceed with an external draft chain (non-DFlash
/// proposer or DFlash cursor reconciled cleanly).
pub(super) fn reconcile_dflash_before_external_draft(model: &dyn Model, a: &mut ActiveSeq) -> bool {
    if !model.proposer_is_dflash() {
        return true;
    }
    match model.reconcile_dflash_context_before_bootstrap(&mut a.seq, 0) {
        Ok(_) => true,
        Err(e) => {
            tracing::error!(
                error = %format!("{e:#}"),
                target_position = a.seq.seq_len,
                "DFLASH_CURSOR_RECONCILE_FAILED_ON_SKIPPED_PROPOSE"
            );
            false
        }
    }
}

fn install_bootstrap_external_draft(
    model: &dyn Model,
    a: &mut ActiveSeq,
    origin: DraftOrigin,
    chain: Vec<u32>,
) -> bool {
    if !reconcile_dflash_before_external_draft(model, a) {
        return false;
    }
    notify_qwen4_external_draft(a, &chain);
    let installed = proposal_lifecycle::install_external_flat(model, a, chain);
    debug_assert!(installed, "a proposed retrieval chain is never empty");
    if installed {
        a.draft_origin = origin;
    }
    installed
}

/// Fill empty/suppressed draft proposals or suffix-graft onto short flat chains
/// up to `num_drafts` using the 65,536-token cross-request validated history ring.
pub(super) fn apply_history_fallback_or_graft(
    model: &dyn Model,
    a: &mut ActiveSeq,
    last_tok: u32,
    num_drafts: usize,
    grammar_masked: bool,
) {
    if num_drafts == 0
        || grammar_masked
        || a.finished
        || a.grammar_state.is_some()
        || a.pending_tree_payload.is_some()
        || a.pending_dflash_sampled_proposal.is_some()
    {
        return;
    }
    if a.pending_drafts.len() >= num_drafts {
        return;
    }
    let Some(extra) = history_ring::fill_or_graft_drafts(
        a.session_hash,
        &a.seq.tokens,
        last_tok,
        &a.pending_drafts,
        a.output_tokens.len(),
        num_drafts,
    ) else {
        return;
    };
    if extra.is_empty() {
        return;
    }
    if a.pending_drafts.is_empty() {
        if !reconcile_dflash_before_external_draft(model, a) {
            return;
        }
        notify_qwen4_external_draft(a, &extra);
        let installed = proposal_lifecycle::install_external_flat(model, a, extra);
        if installed {
            a.draft_origin = DraftOrigin::Proposer;
        }
    } else {
        notify_qwen4_grafted_suffix(a, &extra);
        super::stochastic_verify_walk::truncate_seq_rejection_q(a, a.pending_drafts.len());
        a.pending_drafts.extend(extra);
    }
}

#[inline]
fn resolve_seq_drafts_with(
    proposer_is_dflash: bool,
    depth_policy: verify_qwen4_batched_step::DraftDepthPolicy,
    num_drafts: usize,
    adaptive_seq_depth: impl FnOnce() -> usize,
) -> usize {
    if proposer_is_dflash {
        num_drafts
    } else {
        match depth_policy {
            verify_qwen4_batched_step::DraftDepthPolicy::CohortUniform => num_drafts,
            verify_qwen4_batched_step::DraftDepthPolicy::PerSequence => adaptive_seq_depth(),
        }
    }
}

/// Resolve one sequence's draft budget. DFlash emits its full block in a single
/// forward pass, so MTP `adaptive_depth` never truncates DFlash's configured
/// `gamma`.
#[inline]
fn resolve_seq_drafts(
    model: &dyn Model,
    depth_policy: verify_qwen4_batched_step::DraftDepthPolicy,
    depth_key: u64,
    num_drafts: usize,
) -> usize {
    resolve_seq_drafts_with(model.proposer_is_dflash(), depth_policy, num_drafts, || {
        depth_policy.resolve(depth_key, num_drafts)
    })
}

/// History fallback and suffix grafting only apply on singleton cohorts
/// (`live_seqs == 1` / `PerSequence`), keeping `HISTORY_DRAFTS=1` safe at
/// `c=1` on `C=16` servers without splintering multi-sequence batched widths.
#[inline]
fn history_graft_allowed(
    live_seqs: usize,
    depth_policy: verify_qwen4_batched_step::DraftDepthPolicy,
) -> bool {
    live_seqs == 1 && depth_policy == verify_qwen4_batched_step::DraftDepthPolicy::PerSequence
}

fn post_verify_sync_and_history_graft(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    num_drafts: usize,
    depth_policy: verify_qwen4_batched_step::DraftDepthPolicy,
) {
    history_ring::sync_active_validated_tokens(active);
    if num_drafts == 0
        || model.dflash_draft_sample_method()
            == spark_model::dflash_sampling::DflashDraftSampleMethod::Probabilistic
    {
        return;
    }
    let live_seqs = active.iter().filter(|a| !a.finished).count();
    if !history_graft_allowed(live_seqs, depth_policy) {
        return;
    }
    for a in active.iter_mut() {
        if a.finished
            || request_cancelled(a)
            || dflash_grammar_skip_propose(model, a)
            || a.grammar_state.is_some()
            || a.pending_tree_payload.is_some()
            || a.pending_dflash_sampled_proposal.is_some()
        {
            continue;
        }
        super::stochastic_verify_walk::publish_seq_gumbel_draft_spec(a);
        let seq_drafts =
            resolve_seq_drafts(model, depth_policy, a.depth_controller_key(), num_drafts);
        let last_tok = a.last_token;
        apply_history_fallback_or_graft(model, a, last_tok, seq_drafts, false);
    }
}

/// Save the accepted bonus hidden state into the buffer consumed by the active
/// proposer. DFlash keys its captured row by the emitted token; native MTP
/// consumes the accepted row index from `mtp_hidden_save`.
pub(super) fn save_hidden_for_active_proposer(
    model: &dyn Model,
    a: &mut ActiveSeq,
    token: u32,
    row: usize,
) -> Result<()> {
    if model.proposer_is_dflash() {
        model.save_hidden_for_dflash(token, &mut a.seq, 0)
    } else {
        model.save_hidden_for_mtp_sequence(row, &a.seq, 0)
    }
}

fn dispatch_verify(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    num_drafts: usize,
    think: &ThinkSpecCtx<'_>,
    depth_policy: verify_qwen4_batched_step::DraftDepthPolicy,
) {
    let depth_key = a.depth_controller_key();
    let _depth_session = super::adaptive_depth::ActiveDepthSessionGuard::new(depth_key);
    let seq_drafts = resolve_seq_drafts(model, depth_policy, depth_key, num_drafts);
    let next_num_drafts = if num_drafts == 0 {
        0
    } else {
        seq_drafts.max(1).min(num_drafts)
    };
    let policy_required = verify_policy_required(a, think);
    let generic_proposer = model.proposer_is_dflash()
        || model.proposer_arm_kind(model.proposer_arm())
            == Some(spark_model::speculative::DraftProposerKind::Qwen4NativeMtp);
    let dispatch = if generic_proposer && !drafts.is_empty() {
        VerifyDispatch::Generic
    } else {
        select_verify_dispatch(
            drafts.len(),
            policy_required,
            a.pending_tree_payload.is_some(),
            a.pending_dflash_sampled_proposal.is_some(),
        )
    };
    if dispatch != VerifyDispatch::Generic {
        // Only the generic verifier consumes tree topology.
        a.pending_tree_payload = None;
        // ...and only it attributes accepted tokens to a retrieval tier. A
        // frame narrowed below the generic width (grammar truncation)
        // leaves the counters alone rather than crediting the next frame's
        // acceptances to retrieval.
        a.draft_origin = DraftOrigin::Proposer;
    }
    match dispatch {
        VerifyDispatch::Generic => {
            step_verify_dflash(model, a, drafts, next_num_drafts, think);
        }
        VerifyDispatch::K4 => step_verify_k4(model, a, drafts, next_num_drafts),
        VerifyDispatch::K3 => step_verify_k3(model, a, drafts, next_num_drafts),
        VerifyDispatch::K2 => step_verify_k2(model, a, drafts, next_num_drafts),
        VerifyDispatch::Bootstrap => {
            debug_assert!(drafts.is_empty());
        }
    }
    // Fail-closed exits cannot retain orphaned topology or sampled q.
    clear_orphan_aux(
        &a.pending_drafts,
        &mut a.pending_tree_payload,
        &mut a.pending_dflash_sampled_proposal,
    );
}

fn suppress_finished_verify_cache(
    active: &[ActiveSeq],
    verify_idxs: &[usize],
    cache_on_finish: &mut [bool],
) {
    debug_assert_eq!(active.len(), cache_on_finish.len());
    for &idx in verify_idxs {
        if active[idx].finished {
            cache_on_finish[idx] = false;
        }
    }
}

/// Ask each retrieval tier, in cascade order, for a chain to pre-empt the
/// drafter with.
///
/// Self-context goes first: it is drawn from this sequence's own history,
/// so when it matches at all it matches text the model demonstrably just
/// produced. The static store is the broader, less specific fallback.
/// Both are inert unless their own environment gate is set, and both
/// return a chain of exactly `num_drafts` tokens or nothing.
///
/// `tok` is the token just sampled, or the bonus just accepted. At BOTH
/// propose sites it is the sequence's next token and is not yet in
/// `a.seq.tokens` — the verifier pushes only its INPUT tokens
/// (`verify_d.rs`) — so the committed history is `a.seq.tokens ++ [tok]`
/// and the chain continues from there.
pub(super) fn retrieval_chain(
    a: &mut ActiveSeq,
    tok: u32,
    num_drafts: usize,
) -> Option<(DraftOrigin, Vec<u32>)> {
    if num_drafts < crate::rest_store::MIN_PREEMPT_WIDTH {
        return None;
    }
    // Do not pre-empt a drafter that is currently winning. Retrieval
    // engages on repetitive text, which is exactly where the neural
    // drafter is also strongest — the engine has been observed accepting
    // 15/15 on repetitive parser code. Displacing a frame like that costs
    // the difference, and the offline eval cannot see it because it has
    // no drafter. `last_verify_accepted` can: it is what the drafter just
    // did, on this sequence, on this kind of text.
    let drafter_recent = usize::from(a.last_verify_accepted);
    if crate::rest_store::self_context::enabled()
        && drafter_recent < crate::rest_store::self_context::max_drafter_accept()
    {
        // `a.seq.tokens` is prompt + everything committed so far; `tok`
        // was sampled this step and is not in it yet.
        let chain = a.self_context.propose(&a.seq.tokens, tok, num_drafts);
        if let Some(chain) = chain {
            return Some((DraftOrigin::SelfContext, chain));
        }
    }
    if drafter_recent >= crate::rest_store::max_drafter_accept() {
        return None;
    }
    let chain = crate::rest_store::preempt(&a.seq.tokens, tok, num_drafts)?;
    Some((DraftOrigin::RestStore, chain))
}

/// Predicate gating batched native `K=1` post-prefill bootstrap in Phase A.
///
/// Eligible when at least 2 sequences need bootstrap on a non-EP native Qwen4
/// MTP or non-probabilistic DFlash2 server with BF16 logits; the read-only
/// `Model::qwen4_decode_route` check then confirms `FrameRoute::Native` before
/// submitting `Model::decode_batch`.
#[inline]
#[must_use]
fn batched_native_k1_bootstrap_model_eligible(
    bootstrap_count: usize,
    batched_k5_armed: bool,
    is_ep: bool,
    proposer_is_dflash: bool,
    proposer_is_native_mtp: bool,
    decode_logits_fp32: bool,
    draft_method: spark_model::dflash_sampling::DflashDraftSampleMethod,
) -> bool {
    let proposer_ok = proposer_is_native_mtp || proposer_is_dflash;
    bootstrap_count >= 2
        && batched_k5_armed
        && !is_ep
        && proposer_ok
        && !decode_logits_fp32
        && draft_method != spark_model::dflash_sampling::DflashDraftSampleMethod::Probabilistic
}

/// Run post-prefill `K=1` bootstrap for eligible native Qwen4 MTP or DFlash2
/// sequences in one batched `Model::decode_batch` forward (+ one batched
/// proposal pass) instead of `N` serial `Model::decode` calls.
///
/// Returns the subset of `bootstrap_idxs` that could not batch (singleton,
/// non-native route, multimodal non-identity rotary, or non-Qwen4 model) so
/// the existing Phase A serial loop handles them identically.
fn try_batched_native_k1_bootstrap(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    bootstrap_idxs: &[usize],
    num_drafts: usize,
    think: &ThinkSpecCtx<'_>,
    cache_on_finish: &mut [bool],
) -> Vec<usize> {
    let model_ok = batched_native_k1_bootstrap_model_eligible(
        bootstrap_idxs.len(),
        model.qwen4_batched_k5_armed(),
        model.is_ep(),
        model.proposer_is_dflash(),
        model.proposer_arm_kind(model.proposer_arm())
            == Some(spark_model::speculative::DraftProposerKind::Qwen4NativeMtp),
        model.decode_logits_fp32(),
        model.dflash_draft_sample_method(),
    );
    if !model_ok {
        return bootstrap_idxs.to_vec();
    }

    let mut eligible: Vec<usize> = Vec::with_capacity(bootstrap_idxs.len());
    let mut leftover: Vec<usize> = Vec::new();
    for &idx in bootstrap_idxs {
        let a = &mut active[idx];
        if retire_cancelled_speculation(a, "mtp_before_bootstrap") {
            cache_on_finish[idx] = false;
            continue;
        }
        if a.finished {
            continue;
        }
        if a.seq.rotary_positions.is_identity() {
            eligible.push(idx);
        } else {
            leftover.push(idx);
        }
    }
    if eligible.len() < 2 {
        leftover.extend(eligible);
        return leftover;
    }

    let vocab_size = model.vocab_size();
    let max_rows = spark_model::qwen4_batch_policy::QWEN4_NATIVE_MAX_ROWS;
    let mut propose_ready: Vec<usize> = Vec::with_capacity(eligible.len());

    for chunk in eligible.chunks(max_rows) {
        if chunk.len() < 2 {
            leftover.extend_from_slice(chunk);
            continue;
        }
        let route_is_native = match verify_qwen4_batched_step::batch_sequences_mut(active, chunk)
            .and_then(|seqs| model.qwen4_decode_route(&seqs))
        {
            Ok(Some(spark_model::qwen4_batch_policy::FrameRoute::Native)) => true,
            Ok(_) => false,
            Err(error) => {
                tracing::debug!("batched K=1 bootstrap route check fell back to serial: {error:#}");
                false
            }
        };
        if !route_is_native {
            leftover.extend_from_slice(chunk);
            continue;
        }

        let decode_chunk: Vec<usize> = if model.proposer_is_dflash() {
            let mut ready = Vec::with_capacity(chunk.len());
            for &idx in chunk {
                let a = &mut active[idx];
                match model.reconcile_dflash_context_before_bootstrap(&mut a.seq, 0) {
                    Ok(rows) if rows > 0 => {
                        tracing::info!(
                            rows,
                            target_position = a.seq.seq_len,
                            "DFLASH_CURSOR_RECONCILED"
                        );
                        ready.push(idx);
                    }
                    Ok(_) => ready.push(idx),
                    Err(error) => {
                        tracing::error!(
                            target_position = a.seq.seq_len,
                            "DFLASH_CURSOR_RECONCILE_FAILED: {error:#}"
                        );
                        fail_bootstrap(a, &mut cache_on_finish[idx], "reconcile_context", &error);
                    }
                }
            }
            if ready.len() < 2 {
                leftover.extend(ready);
                continue;
            }
            ready
        } else {
            chunk.to_vec()
        };

        let tokens: Vec<u32> = decode_chunk
            .iter()
            .map(|&idx| active[idx].last_token)
            .collect();
        let decode_result = verify_qwen4_batched_step::batch_sequences_mut(active, &decode_chunk)
            .and_then(|mut seqs| model.decode_batch(&tokens, &mut seqs, 0));
        let logits_base = match decode_result {
            Ok(ptr) => ptr,
            Err(e) => {
                tracing::error!("batched bootstrap decode_batch error: {e:#}");
                for &idx in &decode_chunk {
                    fail_bootstrap(
                        &mut active[idx],
                        &mut cache_on_finish[idx],
                        "decode_batch",
                        &e,
                    );
                }
                continue;
            }
        };

        for (row, &idx) in decode_chunk.iter().enumerate() {
            let a = &mut active[idx];
            let row_logits = logits_base.offset(row * vocab_size * 2);
            let tok = if think.enabled && a.inside_thinking {
                match bootstrap_thinking_token(model, a, row_logits, think) {
                    Some(t) => {
                        if bootstrap_trace_enabled() {
                            tracing::info!(sampled = t, "SPEC_THINK_BOOTSTRAP");
                        }
                        t
                    }
                    None => {
                        fail_bootstrap(
                            a,
                            &mut cache_on_finish[idx],
                            "thinking logits download",
                            "device-to-host copy failed",
                        );
                        continue;
                    }
                }
            } else if !a.inside_thinking && content_policy_required(a, think) {
                match bootstrap_content_token(model, a, row_logits, think) {
                    Some(t) => t,
                    None => {
                        fail_bootstrap(
                            a,
                            &mut cache_on_finish[idx],
                            "content logits download",
                            "device-to-host copy failed",
                        );
                        continue;
                    }
                }
            } else {
                let _gumbel_pos = (super::stochastic_verify_walk::spec_verify_policy()
                    == super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
                    && a.temperature > 0.0)
                    .then(|| {
                        spark_runtime::sampler::gumbel::GumbelPosGuard::new(
                            super::stochastic_verify_walk::resolve_seq_gumbel_seed(a),
                            super::stochastic_verify_walk::seq_abs_pos(a),
                        )
                    });
                let tok = match sample_token_with_grammar(
                    model,
                    row_logits,
                    &a.sampling_params(a.temperature),
                    &[],
                    &a.output_tokens,
                    a.grammar_state.as_mut(),
                ) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!("batched bootstrap sample error: {e:#}");
                        fail_bootstrap(a, &mut cache_on_finish[idx], "sample", &e);
                        continue;
                    }
                };
                let lp = if let Some(k) = a.top_logprobs {
                    extract_single_logprobs(model, row_logits, tok, k)
                } else {
                    None
                };
                if bootstrap_trace_enabled() {
                    tracing::info!(sampled = tok, "SPEC_RAW_BOOTSTRAP");
                }
                emit_token(a, tok, lp);
                tok
            };

            if retire_cancelled_speculation(a, "mtp_after_bootstrap_emit") {
                cache_on_finish[idx] = false;
                continue;
            }
            if a.finished {
                continue;
            }
            a.last_token = tok;
            // Native K=1 batch dispatch already published each sequence's
            // retained target hidden row into `qwen4_mtp_hidden_frames` (and
            // for DFlash, already retired each row's DFlash capture into
            // `DflashProposerState`), so skip `save_hidden_for_mtp_sequence`.
            if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
                tracing::warn!("trim_proposer_state (batched bootstrap reset): {e:#}");
            }
            if retire_cancelled_speculation(a, "mtp_before_proposer") {
                cache_on_finish[idx] = false;
                continue;
            }
            propose_ready.push(idx);
        }
    }

    if !propose_ready.is_empty() {
        verify_qwen4_batched_step::propose_next_batch(model, active, &propose_ready, num_drafts);
        for &idx in &propose_ready {
            let a = &mut active[idx];
            if retire_cancelled_speculation(a, "mtp_after_proposer") {
                cache_on_finish[idx] = false;
                continue;
            }
            if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
                tracing::error!("batched bootstrap start_checkpoint_async: {e:#}");
                fail_bootstrap(a, &mut cache_on_finish[idx], "start_checkpoint_async", e);
            }
        }
    }

    leftover
}

/// Bootstrap empty sequences, then dispatch policy-safe speculative verify.
pub fn step_mtp(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    num_drafts: usize,
    think: &ThinkSpecCtx<'_>,
    cache_on_finish: &mut [bool],
) {
    assert_eq!(active.len(), cache_on_finish.len());
    for a in active.iter_mut() {
        if a.temperature > 0.0
            && a.seed.is_none()
            && spark_runtime::sampler::gumbel::prompt_seed_enabled()
        {
            let prompt_end = a.seq.prompt_len.min(a.seq.tokens.len());
            a.seed = Some(spark_runtime::sampler::seed_from_prompt_tokens(
                &a.seq.tokens[..prompt_end],
            ));
        }
    }
    history_ring::sync_active_validated_tokens(active);
    // ATLAS_QWEN4_BATCHED_ADAPTIVE_K / ATLAS_QWEN4_BATCHED_K5_LADDER (opt-in):
    // narrower drafts at higher concurrency. Lossless; unset keeps
    // `num_drafts` unchanged.
    let live_seqs = active.iter().filter(|a| !a.finished).count();
    let _mtp_dloop_guard = spark_model::layers::mtp_dloop::MtpDloopCohortGuard::enter(
        live_seqs,
        model.mtp_dloop_max_verify_k(),
    );
    let any_sampled = active.iter().any(|a| !a.finished && a.temperature > 0.0);
    verify_qwen4_batched_step::set_step_cohort_sampled(any_sampled);
    let depth_policy = verify_qwen4_batched_step::DraftDepthPolicy::for_cohort(
        model.qwen4_batched_k5_armed(),
        live_seqs,
    );
    let num_drafts = verify_qwen4_batched_step::effective_num_drafts(model, live_seqs, num_drafts);
    // Stage-1 DFlash grammar gate: drop drafts proposed before the grammar
    // became constraining (e.g. the block whose emission opened a
    // `<tool_call>`). DFlash drafts and the K=γ verify argmax bypass the
    // XGrammar bitmask, so a grammar-active sequence must run the bootstrap
    // path below, where `sample_token_with_grammar` enforces it.
    let mut collected_async_idxs: Vec<usize> = Vec::new();
    for (idx, a) in active.iter_mut().enumerate() {
        // ATLAS_DFLASH_ASYNC (task #20): the previous step's propose was
        // ENQUEUED on a second CUDA stream and returned a placeholder chain;
        // its drafter kernels overlapped the SSM commit tail + step-tail CPU
        // work. Collect the real drafts NOW (stream sync + γ×4B D2H) before
        // any gate below inspects `pending_drafts`. `Some(vec![])` = the
        // launch was orphaned → clear and bootstrap (lossless: drafts only
        // ever propose). No-op `None` on the sync path / flag off.
        if !a.pending_drafts.is_empty() {
            match model.dflash_collect_async_drafts(&mut a.seq) {
                Ok(Some(drafts)) => {
                    // ASYNC+DDTree: `collect_async_drafts_impl` may have built a
                    // tree payload from the previous step's top-K kernel. Drain
                    // it now so this step's verify (which checks
                    // `a.pending_tree_payload` inside `step_verify_dflash`) can
                    // use the tree. The normal sync-path drain at the end of
                    // `step_verify_dflash` (`take_pending_tree_payload`) runs
                    // after `run_mtp_propose_multi`, which on the async path
                    // clears `dstate.pending_tree_payload` — so that later drain
                    // returns `None` and does not overwrite the tree set here.
                    let had_drafts = !drafts.is_empty();
                    install_collected(model, a, Ok(drafts))
                        .expect("an infallible collected proposal cannot fail");
                    if had_drafts {
                        collected_async_idxs.push(idx);
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::error!("dflash_collect_async_drafts: {e:#}");
                    proposal_lifecycle::clear_seq(a);
                }
            }
        }
        if !a.pending_drafts.is_empty() && dflash_grammar_skip_propose(model, a) {
            proposal_lifecycle::clear_seq(a);
        }
        // ATLAS_THINK_SPEC=1: the thinking accept filter (dflash_thinking_accept)
        // works on flat token chains only — it compares drafts[i] vs target[i]
        // linearly, but DDTree compact slots are BFS siblings, not descendants,
        // so passing a tree payload through yields at most 1 accepted token (the
        // depth-0 child) before the walk diverges. That is no better than
        // bootstrap but costs a k=32 verify.
        //
        // When inside thinking with a tree payload, drop the tree metadata but
        // KEEP the flat-chain pending_drafts (the top-1 drafter spine). The
        // sequence then falls through to the flat k=17 verify → dflash_thinking_accept,
        // which correctly handles budget accounting, EOS suppression, and the
        // </think> fence. Previously both were cleared → bootstrap (1 tok/step),
        // which caused the Arm C code_long slow-run regression.
        if think.enabled && a.inside_thinking && a.pending_tree_payload.is_some() {
            a.pending_tree_payload = None;
        }
        // A tree walk is indexed by raw verify argmaxes. When the serving
        // sampler has any active policy (Qwen defaults include presence/LZ/DRY
        // penalties), or a post-think mask cannot prove every tree row safe,
        // correctness requires a linear row-by-row policy walk. Keep the
        // drafter's top-1 spine but discard the sibling topology.
        if !a.inside_thinking
            && content_policy_required(a, think)
            && a.pending_tree_payload.is_some()
        {
            if tree_content_raw_argmax_eligible(a, think) {
                static TREE_POLICY_KEEP_DBG: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let n = TREE_POLICY_KEEP_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 3 {
                    tracing::info!("DDTree policy gate: keeping neutral-greedy tree");
                }
            } else {
                static TREE_POLICY_DROP_DBG: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let n = TREE_POLICY_DROP_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 3 {
                    tracing::info!(
                        "DDTree policy gate: dropping tree; using flat sampler-policy walk"
                    );
                }
                a.pending_tree_payload = None;
            }
        }
        // EP synchronizes flat variable-width token frames, but DDTree parent
        // topology/branch compaction is not part of that protocol. Preserve
        // the proposer's top-1 spine and fail closed to flat verification.
        if model.is_ep() && a.pending_tree_payload.is_some() {
            a.pending_tree_payload = None;
        }
        clear_orphan_aux(
            &a.pending_drafts,
            &mut a.pending_tree_payload,
            &mut a.pending_dflash_sampled_proposal,
        );
    }
    if !collected_async_idxs.is_empty() && model.proposer_is_dflash() {
        verify_qwen4_batched_step::finalize_collected_async_dflash_drafts(
            model,
            active,
            &collected_async_idxs,
        );
    }

    // Clamp flat pending drafts to the model's physical KV ceiling before
    // partitioning so sequences approaching `max_seq_len` narrow cleanly or
    // drop to 1-token bootstrap instead of aborting a batched verify cohort.
    if active.iter().any(|a| !a.pending_drafts.is_empty()) {
        let clamp_idxs: Vec<usize> = (0..active.len()).collect();
        verify_qwen4_batched_step::clamp_pending_drafts_to_model_max_seq_len(
            model,
            active,
            &clamp_idxs,
        );
    }

    let mut bootstrap_idxs: Vec<usize> = Vec::new();
    let mut verify_idxs: Vec<usize> = Vec::new();
    for (i, a) in active.iter_mut().enumerate() {
        if retire_cancelled_speculation(a, "mtp_partition") {
            cache_on_finish[i] = false;
            continue;
        }
        match mtp_work(a.finished, !a.pending_drafts.is_empty()) {
            Some(MtpWork::Bootstrap) => bootstrap_idxs.push(i),
            Some(MtpWork::Verify) => verify_idxs.push(i),
            None => {}
        }
    }

    // ── Phase A: Bootstrap decode for sequences without a draft ──
    // A previous verify may have committed its live SSM state asynchronously
    // and then intentionally left drafts empty (grammar transition, proposer
    // miss, diagnostic arm, or error recovery).  Every verify entry waits for
    // that restore; the bootstrap edge must do the same before `decode` reads
    // h_state/conv_state.  Missing this wait made the next token depend on GPU
    // timing and looked like a corrupt partial-accept intermediate.
    if !bootstrap_idxs.is_empty()
        && let Err(e) = model.sync_secondary()
    {
        tracing::error!("sync_secondary before MTP bootstrap: {e:#}");
        for &idx in &bootstrap_idxs {
            fail_bootstrap(
                &mut active[idx],
                &mut cache_on_finish[idx],
                "sync_secondary",
                &e,
            );
        }
        return;
    }
    let serial_bootstrap_idxs = try_batched_native_k1_bootstrap(
        model,
        active,
        &bootstrap_idxs,
        num_drafts,
        think,
        cache_on_finish,
    );
    for &idx in &serial_bootstrap_idxs {
        let a = &mut active[idx];
        if retire_cancelled_speculation(a, "mtp_before_bootstrap") {
            cache_on_finish[idx] = false;
            continue;
        }
        // A verify commits root + accepted drafts before it knows whether the
        // next proposal is policy-safe.  If a thinking/structural boundary
        // suppresses that proposal, its captured hidden rows are still debt
        // against the DFlash ring.  Retire them now, after the secondary
        // commit stream is synchronized and before `decode` overwrites capture
        // row zero.  The model checks the absolute cursor and exact row count;
        // a hole is fatal for this sequence rather than silently realigned.
        match model.reconcile_dflash_context_before_bootstrap(&mut a.seq, 0) {
            Ok(rows) if rows > 0 => tracing::info!(
                rows,
                target_position = a.seq.seq_len,
                "DFLASH_CURSOR_RECONCILED"
            ),
            Ok(_) => {}
            Err(error) => {
                tracing::error!(
                    target_position = a.seq.seq_len,
                    "DFLASH_CURSOR_RECONCILE_FAILED: {error:#}"
                );
                fail_bootstrap(a, &mut cache_on_finish[idx], "reconcile_context", &error);
                continue;
            }
        }
        // EP: broadcast token to worker before decode (worker runs decode in lockstep).
        if let Err(e) = model.ep_broadcast_cmd(a.last_token) {
            tracing::error!("EP broadcast bootstrap token: {e:#}");
            fail_bootstrap(a, &mut cache_on_finish[idx], "EP broadcast", &e);
            continue;
        }
        let logits = match model.decode(a.last_token, &mut a.seq, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("bootstrap decode error: {e:#}");
                fail_bootstrap(a, &mut cache_on_finish[idx], "decode", &e);
                continue;
            }
        };
        let output_len_before_bootstrap = a.output_tokens.len();
        let tok = if think.enabled && a.inside_thinking {
            // ATLAS_THINK_SPEC=1: a thinking sequence's bootstrap token
            // must carry the plain path's logit interventions + per-token
            // side effects (F1/F2/wave, EOS suppression, fence parity),
            // which `sample_token_with_grammar` + `emit_token` do not
            // replicate. `bootstrap_thinking_token` routes this single
            // row through `process_seq_logits` — byte-identical to one
            // `step_decode_only` token — and commits it. The plain
            // decode's per-layer DFlash capture hook already ran inside
            // `model.decode` above, so drafter ctx conditioning matches
            // the propose/verify steps that follow.
            match bootstrap_thinking_token(model, a, logits, think) {
                Some(t) => {
                    if bootstrap_trace_enabled() {
                        tracing::info!(sampled = t, "SPEC_THINK_BOOTSTRAP");
                    }
                    t
                }
                None => {
                    // This helper returns None only on a logits D2H failure;
                    // normal thinking/structural stops still return Some(tok).
                    fail_bootstrap(
                        a,
                        &mut cache_on_finish[idx],
                        "thinking logits download",
                        "device-to-host copy failed",
                    );
                    continue;
                }
            }
        } else if !a.inside_thinking && content_policy_required(a, think) {
            // DFlash bootstrap must use the exact same sampler as ordinary
            // decode.  Raw `sample_token_with_grammar` omits history
            // penalties, logit bias, think masks, and adaptive state.
            match bootstrap_content_token(model, a, logits, think) {
                Some(t) => t,
                None => {
                    // Content stops also return Some(tok); None is D2H error.
                    fail_bootstrap(
                        a,
                        &mut cache_on_finish[idx],
                        "content logits download",
                        "device-to-host copy failed",
                    );
                    continue;
                }
            }
        } else {
            let _gumbel_pos = (super::stochastic_verify_walk::spec_verify_policy()
                == super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
                && a.temperature > 0.0)
                .then(|| {
                    spark_runtime::sampler::gumbel::GumbelPosGuard::new(
                        super::stochastic_verify_walk::resolve_seq_gumbel_seed(a),
                        super::stochastic_verify_walk::seq_abs_pos(a),
                    )
                });
            let tok = match sample_token_with_grammar(
                model,
                logits,
                &a.sampling_params(a.temperature),
                &[],
                &a.output_tokens,
                a.grammar_state.as_mut(),
            ) {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!("bootstrap sample error: {e:#}");
                    fail_bootstrap(a, &mut cache_on_finish[idx], "sample", &e);
                    continue;
                }
            };

            // Extract logprobs from bootstrap decode logits (single position).
            let lp = if let Some(k) = a.top_logprobs {
                extract_single_logprobs(model, logits, tok, k)
            } else {
                None
            };

            if bootstrap_trace_enabled() {
                tracing::info!(sampled = tok, "SPEC_RAW_BOOTSTRAP");
            }

            emit_token(a, tok, lp);
            tok
        };
        if retire_cancelled_speculation(a, "mtp_after_bootstrap_emit") {
            cache_on_finish[idx] = false;
            continue;
        }
        if a.finished {
            if sampled_bootstrap_plain_structural_boundary(
                model.dflash_draft_sample_method()
                    == spark_model::dflash_sampling::DflashDraftSampleMethod::Probabilistic,
                a.finished,
                output_len_before_bootstrap,
                &a.output_tokens,
                tok,
                &a.eos_tokens,
                im_start_hard_stop(),
            ) {
                // This is one terminal-token receipt, not a proposer receipt.
                // Request-bracketed validation combines it with any earlier
                // proposer attempts to distinguish no-proposal from
                // attempted-no-commit structural stops. Invoking the proposer
                // here merely to manufacture an attempt would mutate state
                // after a terminal boundary.
                tracing::warn!("sampled DFlash bootstrap stayed plain: PlainStructuralBoundary");
            }
            continue;
        }
        a.last_token = tok;

        if let Err(e) = model.save_hidden_for_mtp_sequence(0, &a.seq, 0) {
            tracing::error!("save_hidden_for_mtp: {e:#}");
            continue;
        }
        // The pre-decode reconciliation retired any old verify captures and
        // cleared their acceptance metadata. This decode produced exactly one
        // new capture row, so stamp a one-row debt before proposing: the
        // proposal appends row zero at the now-current absolute cursor.
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
            tracing::warn!("trim_proposer_state (bootstrap dflash reset): {e:#}");
        }
        // Cancellation can race after the emitted bootstrap token and while
        // hidden/proposer state is being prepared. Never start a proposal for
        // a request whose client has already gone away.
        if retire_cancelled_speculation(a, "mtp_before_proposer") {
            cache_on_finish[idx] = false;
            continue;
        }
        // Probabilistic DFlash owns a stricter request -> proposal boundary:
        // no REST/PLD/tree/grammar shortcut may publish drafts without their
        // exact sparse-q frame. An ineligible request simply remains on the
        // ordinary one-token bootstrap route.
        if model.dflash_draft_sample_method()
            == spark_model::dflash_sampling::DflashDraftSampleMethod::Probabilistic
        {
            match propose_sampled_dflash(model, a, num_drafts, think) {
                Ok(SampledProposalOutcome::Installed) => tracing::debug!(
                    "sampled DFlash bootstrap: tok={tok} -> drafts={:?}",
                    a.pending_drafts
                ),
                Ok(SampledProposalOutcome::Cancelled) => {
                    cache_on_finish[idx] = false;
                }
                Ok(SampledProposalOutcome::PlainTemperatureZero) => {}
                // This describes the proposal result, not the request's
                // terminal disposition. Keep it distinct from the one
                // PlainStructuralBoundary terminal receipt emitted above so
                // request-bracketed validation cannot double-count one stop.
                Ok(other) => tracing::warn!("sampled DFlash proposal stayed plain: {other:?}"),
                Err(error) => {
                    tracing::error!("sampled DFlash bootstrap held: {error:#}");
                    proposal_lifecycle::clear_seq(a);
                }
            }
        // Stage-1 DFlash grammar gate: while the grammar constrains output,
        // stay non-speculative (the bootstrap decode above already sampled
        // through the grammar; drafting would bypass it at verify).
        } else if !dflash_grammar_skip_propose(model, a) {
            // ── REST conditional pre-emption (default off) ──
            //
            // When the retrieval store holds a long enough verbatim match
            // for this context, its continuation replaces the drafter's
            // chain for THIS step only — the drafter forward pass is then
            // skipped, which is where the second half of the win comes
            // from. REST never displaces DFlash: `preempt` declines unless
            // the match and the resulting chain clear gates set well above
            // break-even, and every decline falls through to the unchanged
            // proposal below. `ATLAS_REST_STORE` unset ⇒ always `None`.
            //
            // Retiring the bootstrap capture row via
            // `reconcile_dflash_before_external_draft` before installing an
            // external chain keeps `dstate.ctx_len` in lockstep with
            // `a.seq.seq_len`, so the upcoming verify never overwrites an
            // unretired capture and trips the fail-closed debt check.
            // Verification is untouched, so emitted tokens are unchanged
            // either way.
            //
            // Cascade order: self-context first — it is the tier most
            // specific to THIS sequence and needs no corpus — then the
            // static store, then the drafter. Each tier declines cheaply
            // when its own env gate is unset.
            let num_drafts = verify_qwen4_batched_step::dflash_gumbel_num_drafts(
                model,
                a.temperature.is_finite() && a.temperature > 0.0,
                live_seqs,
                num_drafts,
            );
            let depth_key = a.depth_controller_key();
            let seq_drafts = resolve_seq_drafts(model, depth_policy, depth_key, num_drafts);
            let allow_history_graft = history_graft_allowed(live_seqs, depth_policy);
            let _depth_session = super::adaptive_depth::ActiveDepthSessionGuard::new(depth_key);
            super::stochastic_verify_walk::publish_seq_gumbel_draft_spec(a);
            history_ring::sync_one_validated_tokens(a.session_hash, &a.output_tokens);
            let retrieved = retrieval_chain(a, tok, num_drafts).or_else(|| {
                (allow_history_graft && a.grammar_state.is_none())
                    .then(|| {
                        history_ring::try_explicit_preempt(
                            a.session_hash,
                            &a.seq.tokens,
                            tok,
                            a.output_tokens.len(),
                            a.last_verify_accepted,
                            num_drafts,
                        )
                        .map(|chain| (DraftOrigin::Proposer, chain))
                    })
                    .flatten()
            });
            if let Some((origin, chain)) = retrieved {
                if install_bootstrap_external_draft(model, a, origin, chain) {
                    tracing::debug!(
                        "retrieval bootstrap ({origin:?}): tok={tok} → drafts={:?}",
                        a.pending_drafts
                    );
                }
            } else {
                let _mtp_grammar_mask = mtp_grammar_mask_for(a);
                match propose_and_install(model, a, tok, num_drafts, _mtp_grammar_mask.as_deref()) {
                    Ok(true) => {
                        if verify_dflash_step::dflash_pcum_enabled_for_seq(a) {
                            verify_qwen4_batched_step::apply_dflash_pcum_truncation(
                                a,
                                model.proposer_is_dflash(),
                            );
                        }
                        tracing::debug!("MTP bootstrap: tok={tok} → drafts={:?}", a.pending_drafts);
                    }
                    Ok(false) => tracing::debug!(tok, num_drafts, "MTP propose returned empty"),
                    Err(e) => tracing::error!("run_mtp_propose_multi: {e:#}"),
                }
                if seq_drafts < a.pending_drafts.len()
                    && a.pending_tree_payload.is_none()
                    && a.pending_dflash_sampled_proposal.is_none()
                {
                    a.pending_drafts.truncate(seq_drafts);
                    super::stochastic_verify_walk::truncate_seq_rejection_q(
                        a,
                        a.pending_drafts.len(),
                    );
                }
                if allow_history_graft {
                    apply_history_fallback_or_graft(
                        model,
                        a,
                        tok,
                        seq_drafts,
                        _mtp_grammar_mask.is_some(),
                    );
                }
            }
        } else {
            // Grammar gate skipped proposal after bootstrap decode: retire the
            // 1-row capture immediately while `dflash_hidden_save[0]` still
            // belongs to this sequence, before any sibling sequence in this
            // tick runs decode or verify.
            let _ = reconcile_dflash_before_external_draft(model, a);
        }

        // Close the last race between the proposer's internal transaction
        // boundary and scheduler checkpointing/retirement.
        if retire_cancelled_speculation(a, "mtp_after_proposer") {
            cache_on_finish[idx] = false;
            continue;
        }

        if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
            tracing::error!("bootstrap start_checkpoint_async: {e:#}");
            fail_bootstrap(a, &mut cache_on_finish[idx], "start_checkpoint_async", e);
        }
    }

    // A request may be cancelled while another sequence is bootstrapping.
    // Re-check the verify cohort before any batched or per-sequence verifier
    // can consume its proposal frame.
    verify_idxs.retain(|&idx| {
        if retire_cancelled_speculation(&mut active[idx], "mtp_before_verify") {
            cache_on_finish[idx] = false;
            false
        } else {
            true
        }
    });

    // ── Phase B (opt-in): CROSS-SEQUENCE BATCHED NATIVE-MTP VERIFY ──
    // ATLAS_QWEN4_BATCHED_K5=1 with per-slot verify lanes provisioned at
    // model construction: neutral greedy flat chains are grouped by width
    // and verified C sequences per target forward. Ineligible sequences,
    // singleton widths and frames refused at the model's read-only admission
    // continue below exactly as before.
    // ATLAS_QWEN4_BATCHED_ADAPTIVE_K: a chain proposed at an earlier,
    // wider concurrency step is shortened to this step's width first.
    if model.qwen4_batched_k5_armed() && verify_qwen4_batched_step::adaptive_selected() {
        verify_qwen4_batched_step::narrow_pending_drafts(active, &verify_idxs, num_drafts);
        verify_qwen4_batched_step::harmonize_cohort_widths(
            model,
            active,
            &verify_idxs,
            num_drafts,
            think,
        );
    }
    if model.qwen4_batched_k5_armed() && verify_idxs.len() >= 2 {
        let plan = verify_qwen4_batched_step::plan_for_cohort(
            model,
            active,
            &verify_idxs,
            num_drafts,
            think,
        );
        let consumed = verify_qwen4_batched_step::step_verify_qwen4_batched(
            model,
            active,
            &plan.frames,
            num_drafts,
            think,
        );
        let mut remaining = plan.leftover;
        for frame in &plan.frames {
            if consumed {
                suppress_finished_verify_cache(active, frame, cache_on_finish);
            } else {
                remaining.extend_from_slice(frame);
            }
        }
        remaining.sort_unstable();
        verify_idxs = remaining;
    }

    // ── Phase B (Path B opt-in, K=2): c-batched K=2 verify when all seqs eligible ──
    // ATLAS_MTP_K2_BATCH_CSEQ=1: K=2 sibling of the K=3 CSK path below.
    // Production `--num-drafts 1` runs the K=2 path which previously
    // never aggregated at concurrency≥2 (each seq ran an independent
    // single-seq graphed K=2 verify, sequentially, on the scheduler
    // thread). This routes c eligible seqs (drafts.len()==1, no
    // grammar, not finished) through ONE c-batched K=2 verify call —
    // 2 batched forwards/layer instead of 2c.
    if spark_model::layers::mtp_k2_batch_cseq_enabled() && verify_idxs.len() >= 2 {
        let mut batched_idxs: Vec<usize> = Vec::new();
        let mut leftover_idxs: Vec<usize> = Vec::new();
        for &idx in &verify_idxs {
            let a = &active[idx];
            let eligibility = proposal_lifecycle::FixedBatchEligibility {
                draft_len: a.pending_drafts.len(),
                grammar_active: a.grammar_state.is_some(),
                finished: a.finished,
                configured_num_drafts: num_drafts,
                policy_required: verify_policy_required(a, think),
                has_tree: a.pending_tree_payload.is_some(),
                has_sampled: a.pending_dflash_sampled_proposal.is_some(),
            };
            if proposal_lifecycle::fixed_batch_decision(
                proposal_lifecycle::FixedBatchWidth::K2,
                eligibility,
            )
            .is_eligible()
            {
                batched_idxs.push(idx);
            } else {
                leftover_idxs.push(idx);
            }
        }
        if batched_idxs.len() >= 2 {
            step_verify_k2_batched_csk(model, active, &batched_idxs, num_drafts);
        } else {
            leftover_idxs.extend(batched_idxs);
        }
        // Process anything that didn't go through the batched path.
        for &idx in &leftover_idxs {
            let a = &mut active[idx];
            let mut drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
            if drafts.is_empty() {
                continue;
            }
            // Thinking spans never advance the grammar (mirrors the
            // `!inside_thinking` gate in emit_token / the bitmask-skip in
            // process_seq_logits), so draft validation against the
            // matcher would spuriously truncate to zero mid-`<think>`.
            // Reachable while thinking under default native-MTP thinking
            // speculation (`ATLAS_MTP_THINK_SPEC`, default ON) or DFlash
            // thinking speculation (`ATLAS_THINK_SPEC=1`).
            if !a.inside_thinking
                && let Some(ref mut gs) = a.grammar_state
            {
                let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
                if !retain_prefix(
                    &mut drafts,
                    &mut a.pending_tree_payload,
                    &mut a.pending_dflash_sampled_proposal,
                    kept,
                ) {
                    super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
                    continue;
                }
                super::stochastic_verify_walk::truncate_seq_rejection_q(a, drafts.len());
            }
            dispatch_verify(model, a, &drafts, num_drafts, think, depth_policy);
        }
        suppress_finished_verify_cache(active, &verify_idxs, cache_on_finish);
        post_verify_sync_and_history_graft(model, active, num_drafts, depth_policy);
        return;
    }

    // ── Phase B (Path B opt-in): c-batched K=3 verify when all seqs eligible ──
    // ATLAS_MTP_K3_BATCH_CSEQ=1: routes verify to the c-batched K=3 per-step
    // K-loop (see model/trait_impl/verify_csk.rs) when all verify_idxs
    // have exactly 2 drafts (K=3 path), no grammar boundary, and not
    // finished. Trades c × 3 = 12 forwards/layer (c=4) for 3 batched
    // forwards/layer + per-step intermediate D2D snapshots (~8 ms
    // overhead). Other verify_idxs (K=2/K=4/DFlash/grammar-active) fall
    // through to the per-seq loop below.
    if spark_model::layers::mtp_k3_batch_cseq_enabled() && verify_idxs.len() >= 2 {
        let mut batched_idxs: Vec<usize> = Vec::new();
        let mut leftover_idxs: Vec<usize> = Vec::new();
        for &idx in &verify_idxs {
            let a = &active[idx];
            let eligibility = proposal_lifecycle::FixedBatchEligibility {
                draft_len: a.pending_drafts.len(),
                grammar_active: a.grammar_state.is_some(),
                finished: a.finished,
                configured_num_drafts: num_drafts,
                policy_required: verify_policy_required(a, think),
                has_tree: a.pending_tree_payload.is_some(),
                has_sampled: a.pending_dflash_sampled_proposal.is_some(),
            };
            if proposal_lifecycle::fixed_batch_decision(
                proposal_lifecycle::FixedBatchWidth::K3,
                eligibility,
            )
            .is_eligible()
            {
                batched_idxs.push(idx);
            } else {
                leftover_idxs.push(idx);
            }
        }
        if batched_idxs.len() >= 2 {
            step_verify_k3_batched_csk(model, active, &batched_idxs, num_drafts);
        } else {
            leftover_idxs.extend(batched_idxs);
        }
        // Process anything that didn't go through the batched path.
        for &idx in &leftover_idxs {
            let a = &mut active[idx];
            let mut drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
            if drafts.is_empty() {
                continue;
            }
            // Thinking spans never advance the grammar (mirrors the
            // `!inside_thinking` gate in emit_token / the bitmask-skip in
            // process_seq_logits), so draft validation against the
            // matcher would spuriously truncate to zero mid-`<think>`.
            // Reachable while thinking under default native-MTP thinking
            // speculation (`ATLAS_MTP_THINK_SPEC`, default ON) or DFlash
            // thinking speculation (`ATLAS_THINK_SPEC=1`).
            if !a.inside_thinking
                && let Some(ref mut gs) = a.grammar_state
            {
                let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
                if !retain_prefix(
                    &mut drafts,
                    &mut a.pending_tree_payload,
                    &mut a.pending_dflash_sampled_proposal,
                    kept,
                ) {
                    super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
                    continue;
                }
                super::stochastic_verify_walk::truncate_seq_rejection_q(a, drafts.len());
            }
            dispatch_verify(model, a, &drafts, num_drafts, think, depth_policy);
        }
        suppress_finished_verify_cache(active, &verify_idxs, cache_on_finish);
        post_verify_sync_and_history_graft(model, active, num_drafts, depth_policy);
        return;
    }

    // ── Phase B (opt-in): CROSS-SEQ BATCHED DFLASH VERIFY (#39) ──
    // ATLAS_DFLASH_BATCHED_VERIFY=1: route ELIGIBLE γ-block verifies through
    // ONE batched forward (FFN weights read once across c seqs). Eligible =
    // drafts.len()>=4, equal K, no grammar, not thinking, no tree payload /
    // tree-or-portfolio method, not finished. Others fall through per-seq.
    if batched_dflash_verify_allowed(
        dflash_batched_verify_enabled(),
        verify_idxs.len(),
        model.is_ep(),
        model.proposer_is_dflash(),
    ) {
        let mut batched_idxs: Vec<usize> = Vec::new();
        let mut leftover_idxs: Vec<usize> = Vec::new();
        // Reference K = first eligible seq's draft count; the batch must share it.
        let mut batch_k: Option<usize> = None;
        let logits_fp32 = model.decode_logits_fp32();
        // Keep the cohort within one single-pass LM-head slab (<= 128 rows) so
        // post-thinking masking (`dflash_think_mask_verified_at_row`) and the
        // full-step verify CUDA graph always see every sequence's resident
        // BF16 logits row.
        const MAX_BATCHED_VERIFY_ROWS: usize = 128;
        for &idx in &verify_idxs {
            let a = &active[idx];
            let dl = a.pending_drafts.len();
            let thinking = think.enabled && a.inside_thinking;
            let row_budget_ok =
                (batched_idxs.len() + 1).saturating_mul(dl + 1) <= MAX_BATCHED_VERIFY_ROWS;
            let eligible = dl >= 4
                && !a.seq.qwen4_qsa_required
                && !a.finished
                && !thinking
                && batched_dflash_content_raw_argmax_eligible(a, think, logits_fp32)
                && a.pending_tree_payload.is_none()
                && a.pending_dflash_sampled_proposal.is_none()
                && !dflash_tree_method_active()
                && !dflash_portfolio_active()
                && batch_k.is_none_or(|k| k == dl)
                && row_budget_ok;
            if eligible {
                batch_k.get_or_insert(dl);
                batched_idxs.push(idx);
            } else {
                leftover_idxs.push(idx);
            }
        }
        if batched_idxs.len() >= 2 {
            // The batched verifier reports no per-sequence acceptance the
            // retrieval counters could consume, so clear provenance here: under
            // ATLAS_DFLASH_BATCHED_VERIFY=1 (default off) engagement and
            // skipped-drafter counts stay exact and accepted-from-retrieval
            // undercounts rather than crediting the wrong frame.
            for &idx in &batched_idxs {
                active[idx].draft_origin = DraftOrigin::Proposer;
                super::stochastic_verify_walk::invalidate_seq_rejection_q(&active[idx]);
            }
            step_verify_dflash_batched(model, active, &batched_idxs, num_drafts);
        } else {
            leftover_idxs.extend(batched_idxs);
        }
        // Per-seq path for anything that didn't batch.
        for &idx in &leftover_idxs {
            let a = &mut active[idx];
            let mut drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
            if drafts.is_empty() {
                continue;
            }
            if !a.inside_thinking
                && let Some(ref mut gs) = a.grammar_state
            {
                let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
                if !retain_prefix(
                    &mut drafts,
                    &mut a.pending_tree_payload,
                    &mut a.pending_dflash_sampled_proposal,
                    kept,
                ) {
                    super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
                    continue;
                }
                super::stochastic_verify_walk::truncate_seq_rejection_q(a, drafts.len());
            }
            dispatch_verify(model, a, &drafts, num_drafts, think, depth_policy);
        }
        suppress_finished_verify_cache(active, &verify_idxs, cache_on_finish);
        post_verify_sync_and_history_graft(model, active, num_drafts, depth_policy);
        return;
    }

    // ── Phase B (default per-seq path): Verify with pipelined checkpoint ──
    for &idx in &verify_idxs {
        let a = &mut active[idx];
        let mut drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        if drafts.is_empty() {
            continue;
        }

        // Spec-decode boundary awareness (arXiv:2512.15834): when a
        // grammar is active, validate the draft sequence against the
        // matcher and truncate at the first token that crosses a
        // grammar transition. Without this, a draft span that crosses
        // `</function>` (or any other structural boundary) gets
        // accepted by the verifier and emitted, but the post-emit
        // `accept_token` silently fails — desync'ing the grammar
        // from the output stream. Truncating here downgrades K=4 →
        // K=3 → K=2 cleanly.
        //
        // Skipped inside `<think>` (reachable under default native-MTP
        // thinking speculation or DFlash `ATLAS_THINK_SPEC=1`): thinking
        // tokens never advance the grammar, so matcher validation would
        // spuriously truncate to zero.
        if !a.inside_thinking
            && let Some(ref mut gs) = a.grammar_state
        {
            let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
            if !retain_prefix(
                &mut drafts,
                &mut a.pending_tree_payload,
                &mut a.pending_dflash_sampled_proposal,
                kept,
            ) {
                super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
                continue;
            }
            super::stochastic_verify_walk::truncate_seq_rejection_q(a, drafts.len());
        }

        // DFlash γ-block drafters return ≥4 drafts per step (γ=15 for a
        // block-16 checkpoint, whose first row is the known anchor), so
        // wide windows route through `step_verify_dflash`. Neutral MTP windows
        // keep the fixed K2/K3/K4 graphs; short policy-sensitive windows also
        // use the generic path because the fixed paths expose only raw argmax.
        //
        // Fix (2026-05-12): route by `drafts.len()` not `num_drafts`. DFlash
        // produces a full γ-block regardless of `num_drafts`; capping via
        // ATLAS_DFLASH_DRAFT_CAP should not force K=2 verify and discard
        // valid drafts.
        dispatch_verify(model, a, &drafts, num_drafts, think, depth_policy);
    }
    suppress_finished_verify_cache(active, &verify_idxs, cache_on_finish);
    post_verify_sync_and_history_graft(model, active, num_drafts, depth_policy);
}

#[cfg(test)]
mod tests {
    use super::{
        MtpWork, VerifyDispatch, batched_dflash_verify_allowed,
        batched_native_k1_bootstrap_model_eligible, history_graft_allowed, mtp_work,
        resolve_seq_drafts_with, sampled_bootstrap_plain_structural_boundary,
        select_verify_dispatch, verify_qwen4_batched_step::DraftDepthPolicy,
    };

    #[test]
    fn cross_sequence_dflash_batching_requires_dflash_proposer() {
        assert!(batched_dflash_verify_allowed(true, 2, false, true));
        assert!(!batched_dflash_verify_allowed(true, 2, false, false));
        assert!(!batched_dflash_verify_allowed(true, 1, false, true));
        assert!(!batched_dflash_verify_allowed(true, 2, true, true));
        assert!(!batched_dflash_verify_allowed(false, 2, false, true));
    }

    #[test]
    fn specialized_verifiers_route_hidden_save_through_proposer_pairing() {
        let cases = [
            (
                "batched DFlash",
                include_str!("verify_dflash_batched_step.rs"),
                1,
            ),
            ("K3", include_str!("verify_k3_step.rs"), 3),
            ("CSK", include_str!("verify_csk_step.rs"), 3),
        ];
        for (label, source, expected_calls) in cases {
            assert_eq!(
                source.matches("save_hidden_for_active_proposer(").count(),
                expected_calls,
                "{label} must pair every accepted outcome with the active proposer"
            );
            assert!(
                !source.contains("model.save_hidden_for_mtp("),
                "{label} must not hard-code the native-MTP hidden buffer"
            );
            assert!(
                !source.contains("model.save_hidden_for_dflash("),
                "{label} must not hard-code the DFlash hidden buffer"
            );
        }
    }

    #[test]
    fn neutral_short_windows_keep_fixed_verifiers() {
        assert_eq!(
            select_verify_dispatch(1, false, false, false),
            VerifyDispatch::K2
        );
        assert_eq!(
            select_verify_dispatch(2, false, false, false),
            VerifyDispatch::K3
        );
        assert_eq!(
            select_verify_dispatch(3, false, false, false),
            VerifyDispatch::K4
        );
    }

    #[test]
    fn policy_short_windows_use_generic_verifier() {
        for draft_len in 1..=3 {
            assert_eq!(
                select_verify_dispatch(draft_len, true, false, false),
                VerifyDispatch::Generic,
                "draft_len={draft_len}"
            );
        }
    }

    #[test]
    fn empty_bootstraps_and_wide_windows_use_generic_verify() {
        assert_eq!(
            select_verify_dispatch(0, false, false, false),
            VerifyDispatch::Bootstrap
        );
        assert_eq!(
            select_verify_dispatch(0, true, false, false),
            VerifyDispatch::Bootstrap
        );
        for draft_len in 4..=17 {
            assert_eq!(
                select_verify_dispatch(draft_len, false, false, false),
                VerifyDispatch::Generic,
                "draft_len={draft_len}"
            );
        }
    }

    #[test]
    fn tree_bearing_width_one_and_two_use_the_topology_consumer() {
        assert_eq!(
            select_verify_dispatch(1, false, true, false),
            VerifyDispatch::Generic
        );
        assert_eq!(
            select_verify_dispatch(2, false, true, false),
            VerifyDispatch::Generic
        );
    }

    #[test]
    fn sampled_q_frames_use_the_generic_consumer_at_every_width() {
        for draft_len in 1..=17 {
            assert_eq!(
                select_verify_dispatch(draft_len, false, false, true),
                VerifyDispatch::Generic,
                "draft_len={draft_len}"
            );
        }
    }

    #[test]
    fn dflash_capture_debt_is_reconciled_before_bootstrap_decode_and_trim() {
        let source = include_str!("mtp_step.rs");
        let phase_a = source
            .split("// ── Phase A: Bootstrap decode for sequences without a draft ──")
            .nth(1)
            .unwrap()
            .split("// ── Phase B")
            .next()
            .unwrap();
        let sync = phase_a.find("model.sync_secondary()").unwrap();
        let reconcile = phase_a
            .find("model.reconcile_dflash_context_before_bootstrap(")
            .unwrap();
        let decode = phase_a.find("model.decode(a.last_token").unwrap();
        let trim = phase_a
            .find("model.trim_proposer_state(&mut a.seq, 0, 0)")
            .unwrap();
        assert!(sync < reconcile && reconcile < decode && decode < trim);
        assert!(phase_a.contains("DFLASH_CURSOR_RECONCILE_FAILED"));
    }

    #[test]
    fn only_committed_terminal_bootstraps_are_benign_zero_proposal_outcomes() {
        const EOS: u32 = 7;
        const IM_START: u32 = 11;
        const CONTENT: u32 = 13;

        assert!(sampled_bootstrap_plain_structural_boundary(
            true,
            true,
            1,
            &[CONTENT, EOS],
            EOS,
            &[EOS],
            Some(IM_START),
        ));
        assert!(sampled_bootstrap_plain_structural_boundary(
            true,
            true,
            1,
            &[CONTENT, IM_START],
            IM_START,
            &[EOS],
            Some(IM_START),
        ));

        // A non-structural token that merely exhausts max_tokens is length.
        assert!(!sampled_bootstrap_plain_structural_boundary(
            true,
            true,
            1,
            &[CONTENT, CONTENT],
            CONTENT,
            &[EOS],
            Some(IM_START),
        ));
        // Cancellation/errors do not commit the just-sampled token.  Requiring
        // exact +1 growth also rejects an equal-valued token already at tail.
        assert!(!sampled_bootstrap_plain_structural_boundary(
            true,
            true,
            1,
            &[EOS],
            EOS,
            &[EOS],
            Some(IM_START),
        ));
        assert!(!sampled_bootstrap_plain_structural_boundary(
            true,
            false,
            1,
            &[CONTENT, EOS],
            EOS,
            &[EOS],
            Some(IM_START),
        ));
        assert!(!sampled_bootstrap_plain_structural_boundary(
            false,
            true,
            1,
            &[CONTENT, EOS],
            EOS,
            &[EOS],
            Some(IM_START),
        ));
    }

    #[test]
    fn terminal_bootstrap_receipt_precedes_and_bypasses_proposer_dispatch() {
        let source = include_str!("mtp_step.rs");
        let phase_a = source
            .split("// ── Phase A: Bootstrap decode for sequences without a draft ──")
            .nth(1)
            .unwrap()
            .split("// ── Phase B")
            .next()
            .unwrap();
        let terminal = phase_a
            .find("sampled_bootstrap_plain_structural_boundary(")
            .unwrap();
        let terminal_continue = phase_a[terminal..].find("continue;").unwrap() + terminal;
        let save_hidden = phase_a
            .find("model.save_hidden_for_mtp_sequence(0, &a.seq, 0)")
            .unwrap();
        let propose = phase_a.find("propose_sampled_dflash(model").unwrap();
        assert!(terminal < terminal_continue);
        assert!(terminal_continue < save_hidden && save_hidden < propose);
        assert_eq!(
            phase_a
                .matches("sampled DFlash bootstrap stayed plain: PlainStructuralBoundary")
                .count(),
            1,
            "the terminal classification is emitted from exactly one source site"
        );
        assert!(phase_a.contains("sampled DFlash proposal stayed plain: {other:?}"));
    }

    #[test]
    fn finished_sequences_never_reenter_bootstrap_or_verify() {
        assert_eq!(mtp_work(true, false), None);
        assert_eq!(mtp_work(true, true), None);
        assert_eq!(mtp_work(false, false), Some(MtpWork::Bootstrap));
        assert_eq!(mtp_work(false, true), Some(MtpWork::Verify));
    }

    #[test]
    fn dflash_bypasses_per_sequence_adaptive_depth_while_mtp_respects_it() {
        // Under Gumbel/Stochastic, adaptive depth can return a shortened depth (e.g. 2 or 0).
        // DFlash must preserve full gamma=7 without consulting adaptive_seq_depth.
        assert_eq!(
            resolve_seq_drafts_with(true, DraftDepthPolicy::PerSequence, 7, || {
                panic!("DFlash must not invoke per-sequence adaptive depth")
            }),
            7
        );
        assert_eq!(
            resolve_seq_drafts_with(false, DraftDepthPolicy::PerSequence, 7, || 2),
            2
        );
        assert_eq!(
            resolve_seq_drafts_with(false, DraftDepthPolicy::CohortUniform, 4, || 1),
            4
        );
    }

    #[test]
    fn history_graft_only_engages_on_singleton_per_sequence_cohorts() {
        assert!(history_graft_allowed(1, DraftDepthPolicy::PerSequence));
        assert!(!history_graft_allowed(2, DraftDepthPolicy::PerSequence));
        assert!(!history_graft_allowed(1, DraftDepthPolicy::CohortUniform));
        assert!(!history_graft_allowed(4, DraftDepthPolicy::CohortUniform));
        assert!(!history_graft_allowed(0, DraftDepthPolicy::PerSequence));
    }

    #[test]
    fn batched_native_k1_bootstrap_requires_multi_seq_native_mtp_bf16_non_ep() {
        use spark_model::dflash_sampling::DflashDraftSampleMethod;

        assert!(batched_native_k1_bootstrap_model_eligible(
            16,
            true,
            false,
            false,
            true,
            false,
            DflashDraftSampleMethod::Greedy,
        ));
        // Non-probabilistic DFlash2 is also admitted for batched native K=1 bootstrap.
        assert!(batched_native_k1_bootstrap_model_eligible(
            4,
            true,
            false,
            true,
            false,
            false,
            DflashDraftSampleMethod::Greedy,
        ));
        // Singletons stay on the graphed C=1 decode path.
        assert!(!batched_native_k1_bootstrap_model_eligible(
            1,
            true,
            false,
            false,
            true,
            false,
            DflashDraftSampleMethod::Greedy,
        ));
        assert!(!batched_native_k1_bootstrap_model_eligible(
            1,
            true,
            false,
            true,
            false,
            false,
            DflashDraftSampleMethod::Greedy,
        ));
        // EP, FP32 logits, un-armed servers, non-MTP/non-DFlash proposers, and probabilistic DFlash stay serial.
        assert!(!batched_native_k1_bootstrap_model_eligible(
            4,
            false,
            false,
            false,
            true,
            false,
            DflashDraftSampleMethod::Greedy,
        ));
        assert!(!batched_native_k1_bootstrap_model_eligible(
            4,
            true,
            true,
            false,
            true,
            false,
            DflashDraftSampleMethod::Greedy,
        ));
        assert!(!batched_native_k1_bootstrap_model_eligible(
            4,
            true,
            false,
            false,
            false,
            false,
            DflashDraftSampleMethod::Greedy,
        ));
        assert!(!batched_native_k1_bootstrap_model_eligible(
            4,
            true,
            false,
            false,
            true,
            true,
            DflashDraftSampleMethod::Greedy,
        ));
        assert!(!batched_native_k1_bootstrap_model_eligible(
            4,
            true,
            false,
            false,
            true,
            false,
            DflashDraftSampleMethod::Probabilistic,
        ));
        assert!(!batched_native_k1_bootstrap_model_eligible(
            4,
            true,
            false,
            true,
            false,
            false,
            DflashDraftSampleMethod::Probabilistic,
        ));
    }

    mod dflash_cursor_lockstep_tests {
        use std::sync::{
            Mutex,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        };

        use anyhow::Result;
        use spark_model::traits::{Model, SequenceState};
        use spark_runtime::gpu::DevicePtr;

        use super::super::{
            DraftOrigin, ResponseSink, ThinkSpecCtx, apply_history_fallback_or_graft, history_ring,
            install_bootstrap_external_draft, step_mtp,
        };
        use crate::scheduler::test_fixtures::{active_seq, model as test_model};

        struct DflashCursorModel {
            ctx_cursor: Mutex<usize>,
            last_num_accepted: Mutex<usize>,
            reconcile_retired_rows: Mutex<Vec<usize>>,
            propose_errors: AtomicUsize,
            fail_reconcile: AtomicBool,
            propose_returns_empty: AtomicBool,
            verify_targets: Mutex<Vec<u32>>,
            bootstrap_token: u32,
        }

        impl DflashCursorModel {
            fn new(initial_cursor: usize, bootstrap_token: u32, verify_targets: Vec<u32>) -> Self {
                Self {
                    ctx_cursor: Mutex::new(initial_cursor),
                    last_num_accepted: Mutex::new(0),
                    reconcile_retired_rows: Mutex::new(Vec::new()),
                    propose_errors: AtomicUsize::new(0),
                    fail_reconcile: AtomicBool::new(false),
                    propose_returns_empty: AtomicBool::new(false),
                    verify_targets: Mutex::new(verify_targets),
                    bootstrap_token,
                }
            }

            fn cursor(&self) -> usize {
                *self.ctx_cursor.lock().unwrap()
            }

            fn append_pending_captures(&self, position: usize) -> Result<usize> {
                let mut cursor = self.ctx_cursor.lock().unwrap();
                anyhow::ensure!(
                    position >= *cursor,
                    "DFlash context cursor {} is ahead of target position {position}",
                    *cursor,
                );
                let pending_rows = position - *cursor;
                if pending_rows == 0 {
                    return Ok(0);
                }
                let recorded_rows = self
                    .last_num_accepted
                    .lock()
                    .unwrap()
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("DFlash accepted-row count overflowed"))?;
                anyhow::ensure!(
                    pending_rows == recorded_rows,
                    "DFlash context debt disagrees with captured verify rows: \
                     cursor={} position={position} pending={pending_rows} recorded={recorded_rows}",
                    *cursor,
                );
                *cursor = position;
                Ok(pending_rows)
            }
        }

        impl Model for DflashCursorModel {
            fn prefill(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
                unreachable!()
            }
            fn prefill_chunk(
                &self,
                _: &[u32],
                _: &mut SequenceState,
                _: usize,
                _: usize,
                _: bool,
                _: u64,
            ) -> Result<DevicePtr> {
                unreachable!()
            }
            fn decode(&self, token: u32, seq: &mut SequenceState, _: u64) -> Result<DevicePtr> {
                seq.seq_len += 1;
                seq.tokens.push(token);
                Ok(DevicePtr::NULL)
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
                128
            }
            fn bind_gpu_to_thread(&self) -> Result<()> {
                Ok(())
            }
            fn alloc_sequence(&self) -> Result<SequenceState> {
                Ok(test_model::seq_state())
            }
            fn copy_logits_to_host(&self, _: DevicePtr, _: &mut [u8]) -> Result<()> {
                unreachable!()
            }
            fn logits_buffer_ptr(&self) -> DevicePtr {
                DevicePtr::NULL
            }
            fn argmax_on_device(&self, _: DevicePtr, _: u64) -> Result<u32> {
                Ok(self.bootstrap_token)
            }
            fn argmax_batch(&self, _: DevicePtr, _: usize, _: u64) -> Result<Vec<u32>> {
                unreachable!()
            }
            fn hidden_after_norm(&self) -> DevicePtr {
                DevicePtr::NULL
            }
            fn decode_verify(
                &self,
                tokens: &[u32],
                seq: &mut SequenceState,
                _: u64,
            ) -> Result<Vec<u32>> {
                seq.seq_len += tokens.len();
                seq.tokens.extend_from_slice(tokens);
                Ok(self.verify_targets.lock().unwrap().clone())
            }
            fn decode_verify_dflash(
                &self,
                tokens: &[u32],
                seq: &mut SequenceState,
                stream: u64,
            ) -> Result<Vec<u32>> {
                self.decode_verify(tokens, seq, stream)
            }
            fn checkpoint_ssm_states(&self, _: &mut SequenceState) -> Result<()> {
                Ok(())
            }
            fn rollback_ssm_states(&self, _: &mut SequenceState, _: usize) -> Result<()> {
                Ok(())
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
                true
            }
            fn proposer_is_dflash(&self) -> bool {
                true
            }
            fn dflash_verify_capacity_k(&self) -> Option<usize> {
                Some(8)
            }
            fn has_self_speculative(&self) -> bool {
                false
            }
            fn decode_draft(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
                unreachable!()
            }
            fn cache_sequence(&self, _: &SequenceState) {}
            fn free_sequence(&self, _: &mut SequenceState) -> Result<()> {
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
                Ok(())
            }
            fn save_hidden_for_dflash(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<()> {
                Ok(())
            }
            fn reconcile_dflash_context_before_bootstrap(
                &self,
                seq: &mut SequenceState,
                _: u64,
            ) -> Result<usize> {
                if self.fail_reconcile.load(Ordering::SeqCst) {
                    anyhow::bail!("injected DFlash cursor reconcile failure");
                }
                let retired = self.append_pending_captures(seq.seq_len)?;
                *self.last_num_accepted.lock().unwrap() = 0;
                if retired > 0 {
                    self.reconcile_retired_rows.lock().unwrap().push(retired);
                }
                Ok(retired)
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
                position: usize,
                num_drafts: usize,
                _: &mut SequenceState,
                _: u64,
                _: Option<&[i32]>,
            ) -> Result<Vec<u32>> {
                if self.propose_returns_empty.load(Ordering::SeqCst) {
                    return Ok(Vec::new());
                }
                match self.append_pending_captures(position) {
                    Ok(_) => Ok((0..num_drafts).map(|i| 81 + i as u32).collect()),
                    Err(e) => {
                        self.propose_errors.fetch_add(1, Ordering::SeqCst);
                        Err(e)
                    }
                }
            }
            fn trim_proposer_state(
                &self,
                _: &mut SequenceState,
                num_accepted: usize,
                _: u64,
            ) -> Result<()> {
                *self.last_num_accepted.lock().unwrap() = num_accepted;
                Ok(())
            }
            fn start_checkpoint_async(&self, _: &mut SequenceState) -> Result<()> {
                Ok(())
            }
            fn sync_secondary(&self) -> Result<()> {
                Ok(())
            }
            fn commit_verify_state_async(
                &self,
                _: &mut SequenceState,
                _: usize,
                _: usize,
            ) -> Result<()> {
                Ok(())
            }
            fn synchronize_stream(&self, _: u64) -> Result<()> {
                Ok(())
            }
        }

        fn neutral_think_ctx() -> ThinkSpecCtx<'static> {
            ThinkSpecCtx {
                enabled: false,
                code_fence_token: None,
                reflection_suppress_ids: &[],
                adaptive_sampling: false,
                honor_eos_inside_thinking: Some(false),
            }
        }

        fn multiturn_6799_seq(sink: ResponseSink, session_hash: u64) -> super::super::ActiveSeq {
            let mut a = active_seq(sink, 0);
            a.think_ended = false;
            a.session_hash = session_hash;
            a.seq.seq_len = 6_799;
            a.seq.prompt_len = 6_799;
            a.seq.tokens = vec![7; 6_799];
            a.last_token = 31;
            a.eos_tokens = vec![17];
            a
        }

        #[test]
        fn multiturn_6799_bootstrap_external_draft_and_verify_keep_dflash_cursor_in_lockstep() {
            let think = neutral_think_ctx();

            // Case 1: Exact multi-turn TC-88 sequence (6799 -> 6800 -> 6802).
            // Bootstrap decode advances seq_len to 6800, then an external draft
            // chain pre-empts DFlash propose before verify accepts 1 draft.
            let model = DflashCursorModel::new(6_799, 31, vec![41, 77, 90, 91, 92]);
            let (tx, mut rx) = tokio::sync::oneshot::channel();
            let mut a = multiturn_6799_seq(ResponseSink::Blocking(Some(tx)), 0x5443_0088_0001);

            assert_eq!(
                model
                    .reconcile_dflash_context_before_bootstrap(&mut a.seq, 0)
                    .unwrap(),
                0
            );
            model.decode(a.last_token, &mut a.seq, 0).unwrap();
            model.trim_proposer_state(&mut a.seq, 0, 0).unwrap();
            assert_eq!(a.seq.seq_len, 6_800);
            assert_eq!(model.cursor(), 6_799);

            assert!(install_bootstrap_external_draft(
                &model,
                &mut a,
                DraftOrigin::Proposer,
                vec![41, 42, 43, 44],
            ));
            assert_eq!(model.cursor(), 6_800);
            assert_eq!(*model.reconcile_retired_rows.lock().unwrap(), [1]);
            assert_eq!(a.pending_drafts, [41, 42, 43, 44]);

            let mut active = vec![a];
            let mut cache = vec![true];
            step_mtp(&model, &mut active, 7, &think, &mut cache);

            assert!(!active[0].finished);
            assert_eq!(active[0].seq.seq_len, 6_802);
            assert_eq!(model.cursor(), 6_802);
            assert_eq!(active[0].last_token, 77);
            assert_eq!(active[0].pending_drafts, [81, 82, 83, 84, 85, 86, 87]);
            assert_eq!(model.propose_errors.load(Ordering::SeqCst), 0);
            assert_eq!(cache, [true]);
            assert!(rx.try_recv().is_err());

            // Case 2: End-to-end two-tick step_mtp across history fallback/pre-empt
            // at 6799 -> 6800 -> 6802 (covers both FallbackOnly and Explicit modes).
            if history_ring::history_draft_mode() != history_ring::HistoryDraftMode::Off {
                let model = DflashCursorModel::new(6_799, 104, vec![41, 77, 90, 91, 92]);
                model.propose_returns_empty.store(true, Ordering::SeqCst);
                let (tx, mut rx) = tokio::sync::oneshot::channel();
                let mut a = multiturn_6799_seq(ResponseSink::Blocking(Some(tx)), 0x5443_0088_0002);
                a.seq.tokens[6_780..6_788].copy_from_slice(&[101, 102, 103, 104, 41, 42, 43, 44]);
                a.seq.tokens[6_797..6_799].copy_from_slice(&[101, 102]);
                a.last_token = 103;

                let mut active = vec![a];
                let mut cache = vec![true];
                step_mtp(&model, &mut active, 7, &think, &mut cache);

                assert!(!active[0].finished);
                assert_eq!(active[0].seq.seq_len, 6_800);
                assert_eq!(model.cursor(), 6_800);
                assert_eq!(*model.reconcile_retired_rows.lock().unwrap(), [1]);
                assert_eq!(active[0].pending_drafts, [41, 42, 43, 44]);

                model.propose_returns_empty.store(false, Ordering::SeqCst);
                step_mtp(&model, &mut active, 7, &think, &mut cache);

                assert!(!active[0].finished);
                assert_eq!(active[0].seq.seq_len, 6_802);
                assert_eq!(model.cursor(), 6_802);
                assert_eq!(active[0].last_token, 77);
                assert_eq!(active[0].pending_drafts, [81, 82, 83, 84, 85, 86, 87]);
                assert_eq!(model.propose_errors.load(Ordering::SeqCst), 0);
                assert_eq!(cache, [true]);
                assert!(rx.try_recv().is_err());
            }
        }

        #[test]
        fn external_draft_installation_withholds_chain_when_dflash_reconcile_fails() {
            let model = DflashCursorModel::new(6_799, 104, vec![41, 77, 90, 91, 92]);
            model.fail_reconcile.store(true, Ordering::SeqCst);

            let (tx, _rx) = tokio::sync::oneshot::channel();
            let mut a = multiturn_6799_seq(ResponseSink::Blocking(Some(tx)), 0x5443_0088_0003);
            a.seq.tokens[6_780..6_788].copy_from_slice(&[101, 102, 103, 104, 41, 42, 43, 44]);
            a.seq.tokens[6_797..6_799].copy_from_slice(&[101, 102]);
            a.last_token = 103;
            model.decode(a.last_token, &mut a.seq, 0).unwrap();
            model.trim_proposer_state(&mut a.seq, 0, 0).unwrap();

            assert!(!install_bootstrap_external_draft(
                &model,
                &mut a,
                DraftOrigin::Proposer,
                vec![41, 42, 43, 44],
            ));
            assert!(a.pending_drafts.is_empty());
            assert_eq!(model.cursor(), 6_799);

            apply_history_fallback_or_graft(&model, &mut a, 104, 7, false);
            assert!(a.pending_drafts.is_empty());
            assert_eq!(model.cursor(), 6_799);
        }
    }
}

#[cfg(test)]
#[path = "rest_preempt_tests.rs"]
mod rest_preempt_tests;
