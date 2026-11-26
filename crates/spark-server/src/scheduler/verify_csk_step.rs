// SPDX-License-Identifier: AGPL-3.0-only

//! K=3 verify, c-batched per-step (Path B).
//!
//! Routes c verify-eligible sequences (each with `drafts.len() == 2`)
//! through a single c-batched K=3 verify call instead of c independent
//! single-seq `decode_verify_graphed_k3` calls. The trait method
//! `decode_verify_k3_batched_csk` runs K=3 SEQUENTIAL c-batched single-
//! step decodes (each at M=c via the existing `decode_multi_seq` path);
//! this scheduler function decodes the per-seq argmax results, applies
//! per-seq accept/reject + commit + emit + propose using the SAME helpers
//! as the per-seq `step_verify_k3` path.
//!
//! **Gated**: only invoked when `ATLAS_MTP_K3_BATCH_CSEQ=1` AND all
//! verify candidates pass the eligibility filter in `step_mtp`
//! (drafts.len() == 2, grammar inactive, not finished, c ≥ 2).

use super::*;

/// K=3 c-batched verify dispatch. `batched_idxs` indexes into `active`
/// the sequences eligible for batched verify (filtered upstream in
/// `step_mtp`). All listed sequences MUST have `drafts.len() == 2`.
pub fn step_verify_k3_batched_csk(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    batched_idxs: &[usize],
    num_drafts: usize,
) {
    let c = batched_idxs.len();
    if c < 2 {
        return;
    }

    if proposal_lifecycle::flat_batch_preflight_at(active, batched_idxs)
        != proposal_lifecycle::FlatBatchPreflight::Ready
    {
        tracing::error!("CSK-K3 received a tree-bearing proposal; refusing flat batch");
        return;
    }

    if let Err(e) = model.sync_secondary() {
        tracing::error!("CSK sync_secondary: {e:#}");
        for &idx in batched_idxs {
            fail_speculative_step(&mut active[idx], "CSK K3 sync_secondary", &e);
        }
        return;
    }

    // ── Gather per-seq inputs ──
    // `tokens_per_seq[i] = [last_token, drafts[0], drafts[1]]`
    // `drafts[i]` saved here for accept/reject comparison below.
    let mut tokens_per_seq: Vec<[u32; 3]> = Vec::with_capacity(c);
    let mut per_seq_drafts: Vec<[u32; 2]> = Vec::with_capacity(c);
    let mut original_seq_lens: Vec<usize> = Vec::with_capacity(c);

    let Some(batch_drafts) = proposal_lifecycle::take_flat_batch_at(active, batched_idxs) else {
        tracing::error!("CSK-K3 flat batch changed after preflight; refusing batch");
        return;
    };
    for (&idx, drafts) in batched_idxs.iter().zip(batch_drafts) {
        let a = &mut active[idx];
        debug_assert_eq!(drafts.len(), 2, "CSK eligibility guarantees 2 drafts");
        tokens_per_seq.push([a.last_token, drafts[0], drafts[1]]);
        per_seq_drafts.push([drafts[0], drafts[1]]);
        original_seq_lens.push(a.seq.seq_len);
    }

    // EP: c-batched verify_csk is NOT EP-aware. Step_mtp's eligibility
    // filter does not check EP — gate here as a safety net. (Production
    // single-GPU deploy doesn't hit EP, so this is a defensive bail.)
    if model.is_ep() {
        tracing::warn!("CSK verify path doesn't support EP; falling back to per-seq K=3 for c={c}");
        // Restore drafts and fall through to per-seq path.
        for (i, &idx) in batched_idxs.iter().enumerate() {
            active[idx].pending_drafts = per_seq_drafts[i].to_vec();
            let drafts_vec = std::mem::take(&mut active[idx].pending_drafts);
            step_verify_k3(model, &mut active[idx], &drafts_vec, num_drafts);
        }
        return;
    }

    // ── Batched K-loop verify ──
    let t_verify = Instant::now();
    let results = {
        let mut refs: Vec<&mut SequenceState> = Vec::with_capacity(c);
        // Safe disjoint-index borrow: sort indices ascending, peel off each
        // `&mut ActiveSeq` with `split_at_mut`, then permute back to
        // `batched_idxs` order.
        let mut sorted: Vec<(usize, usize)> = batched_idxs
            .iter()
            .enumerate()
            .map(|(p, &i)| (i, p))
            .collect();
        sorted.sort_by_key(|t| t.0);
        let mut tail: &mut [ActiveSeq] = active;
        let mut cursor: usize = 0;
        let mut sorted_refs: Vec<(usize, &mut SequenceState)> = Vec::with_capacity(c);
        for (idx, orig_pos) in &sorted {
            let offset = idx - cursor;
            let (_skip, rest) = tail.split_at_mut(offset);
            let (head, rest_after) = rest.split_first_mut().expect("idx within bounds");
            sorted_refs.push((*orig_pos, &mut head.seq));
            tail = rest_after;
            cursor = idx + 1;
        }
        // Permute back to batched_idxs order.
        sorted_refs.sort_by_key(|(p, _)| *p);
        refs.extend(sorted_refs.into_iter().map(|(_, s)| s));

        // Also need tokens_per_seq in matching order — already constructed
        // in batched_idxs order. refs is now also in batched_idxs order.
        match model.decode_verify_k3_batched_csk(&tokens_per_seq, &mut refs, 0) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("decode_verify_k3_batched_csk: {e:#}");
                drop(refs);
                for &idx in batched_idxs {
                    fail_speculative_step(&mut active[idx], "CSK K3 decode_verify", &e);
                }
                return;
            }
        }
    };
    let verify_us = t_verify.elapsed().as_micros();

    // ── Per-seq accept/reject + commit + emit + propose ──
    for (i, &idx) in batched_idxs.iter().enumerate() {
        let a = &mut active[idx];
        a.last_token_time = Instant::now();
        let [v0, v1, v2] = results[i];
        let drafts = per_seq_drafts[i];
        let orig_len = original_seq_lens[i];

        // Per-seq num_accepted (matches K=3 single-seq logic):
        //   drafts[0] != v0  → 0 drafts accepted, num_accepted=1
        //   drafts[1] != v1  → 1 draft accepted,  num_accepted=2
        //   else (full)      → 2 drafts accepted, num_accepted=3
        let num_accepted_drafts = if drafts[0] != v0 {
            0
        } else if drafts[1] != v1 {
            1
        } else {
            2
        };
        // Note: K=3 single-seq's `num_accepted` is drafts_accepted (0/1/2),
        // and commit_verify_state_async takes `k=3` + `num_accepted ∈
        // {1, 2, 3}` (full-token-count including last_token's accept).
        // Keep this naming consistent with verify_k3_step.rs comments.

        crate::metrics::SPEC_DECODE_VERIFY
            .with_label_values(&[
                "3",
                match num_accepted_drafts {
                    2 => "accept_all",
                    1 => "accept_partial",
                    _ => "reject",
                },
            ])
            .inc();
        crate::metrics::compat_metrics::record_spec_decode(2, num_accepted_drafts as u64);

        // Truncate seq state from K=3 advance to actual acceptance level.
        // decode_batch advanced seq.seq_len by K=3 and pushed 3 tokens.
        // We want `seq_len = orig_len + num_accepted_drafts + 1`.
        let target_len = orig_len + num_accepted_drafts + 1;
        if !super::mod_helpers::truncate_csk_verify_extent(a, orig_len, target_len, 3) {
            continue;
        }

        // Extract logprobs (positions match the verify output buffer
        // populated by per-step argmax). Note: `extract_verify_logprobs`
        // reads from the shared lm_head logits buffer which got
        // overwritten by EACH of the K=3 steps; only the LAST step's
        // logits are still in the buffer. So logprobs are unavailable
        // for K-loop CSK path — disable to avoid emitting wrong data.
        // TODO: per-step D2H of logprobs into per-step scratch.
        let verify_lps: Vec<Option<crate::api::TokenLogprobs>> = Vec::new();

        // Per-seq commit + trim + emit + propose pipeline. Mirrors
        // verify_k3_step.rs branches.
        if num_accepted_drafts == 2 {
            // ── FULL ACCEPT (K=3 accept-3) ──
            let mut consumed_rows = 0usize;
            if emit_token_for_verify(a, drafts[0], verify_lps.first().cloned().flatten()) {
                consumed_rows += 1;
            }
            if !a.finished
                && emit_token_for_verify(a, drafts[1], verify_lps.get(1).cloned().flatten())
            {
                consumed_rows += 1;
            }
            if !a.finished && emit_token_for_verify(a, v2, verify_lps.get(2).cloned().flatten()) {
                consumed_rows += 1;
            }
            if a.finished {
                super::mod_helpers::reconcile_terminal_csk_verify(
                    model,
                    a,
                    orig_len,
                    consumed_rows,
                    3,
                );
                continue;
            }
            a.last_token = v2;

            if let Err(e) = model.commit_verify_state_async(&mut a.seq, 3, 3) {
                tracing::error!("CSK commit_verify_state_async (K=3 accept-3): {e:#}");
                fail_speculative_step(a, "CSK K3 commit accepted prefix", e);
                continue;
            }
            if let Err(e) = save_hidden_for_active_proposer(model, a, v2, 2) {
                tracing::error!("CSK save hidden for active proposer (row 2): {e:#}");
                continue;
            }
            if let Err(e) = model.trim_proposer_state(&mut a.seq, 2, 0) {
                tracing::error!("CSK trim_proposer_state: {e:#}");
            }
            let _mtp_grammar_mask = mtp_grammar_mask_for(a);
            match proposal_lifecycle::propose_and_install(
                model,
                a,
                v2,
                num_drafts,
                _mtp_grammar_mask.as_deref(),
            ) {
                Ok(_) => {}
                Err(e) => tracing::error!("CSK run_mtp_propose_multi: {e:#}"),
            }
        } else if num_accepted_drafts == 1 {
            // ── PARTIAL ACCEPT-2 (last_token + draft[0]) ──
            let mut consumed_rows = 0usize;
            if emit_token_for_verify(a, drafts[0], verify_lps.first().cloned().flatten()) {
                consumed_rows += 1;
            }
            if !a.finished && emit_token_for_verify(a, v1, verify_lps.get(1).cloned().flatten()) {
                consumed_rows += 1;
            }
            if a.finished {
                super::mod_helpers::reconcile_terminal_csk_verify(
                    model,
                    a,
                    orig_len,
                    consumed_rows,
                    3,
                );
                continue;
            }
            a.last_token = v1;
            if let Err(e) = model.trim_proposer_state(&mut a.seq, 1, 0) {
                tracing::error!("CSK trim_proposer_state: {e:#}");
            }
            if let Err(e) = model.commit_verify_state_async(&mut a.seq, 2, 3) {
                tracing::error!("CSK commit_verify_state_async (K=3 accept-2): {e:#}");
                fail_speculative_step(a, "CSK K3 commit partial prefix", e);
                continue;
            }
            if let Err(e) = save_hidden_for_active_proposer(model, a, v1, 1) {
                tracing::error!("CSK save hidden for active proposer (row 1): {e:#}");
                continue;
            }
            let _mtp_grammar_mask = mtp_grammar_mask_for(a);
            match proposal_lifecycle::propose_and_install(
                model,
                a,
                v1,
                num_drafts,
                _mtp_grammar_mask.as_deref(),
            ) {
                Ok(_) => {}
                Err(e) => tracing::error!("CSK run_mtp_propose_multi: {e:#}"),
            }
        } else {
            // ── REJECT (only last_token accepted) ──
            let mut consumed_rows = 0usize;
            if emit_token_for_verify(a, v0, verify_lps.first().cloned().flatten()) {
                consumed_rows += 1;
            }
            if a.finished {
                super::mod_helpers::reconcile_terminal_csk_verify(
                    model,
                    a,
                    orig_len,
                    consumed_rows,
                    3,
                );
                continue;
            }
            a.last_token = v0;
            if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
                tracing::error!("CSK trim_proposer_state: {e:#}");
            }
            if let Err(e) = model.commit_verify_state_async(&mut a.seq, 1, 3) {
                tracing::error!("CSK commit_verify_state_async (K=3 accept-1): {e:#}");
                fail_speculative_step(a, "CSK K3 commit rejected prefix", e);
                continue;
            }
            if let Err(e) = save_hidden_for_active_proposer(model, a, v0, 0) {
                tracing::error!("CSK save hidden for active proposer (row 0): {e:#}");
                continue;
            }
            let _mtp_grammar_mask = mtp_grammar_mask_for(a);
            match proposal_lifecycle::propose_and_install(
                model,
                a,
                v0,
                num_drafts,
                _mtp_grammar_mask.as_deref(),
            ) {
                Ok(_) => {}
                Err(e) => tracing::error!("CSK run_mtp_propose_multi: {e:#}"),
            }
        }
    }

    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());
    tracing::debug!("CSK K=3 verify: c={c} verify={verify_us}μs");
}
