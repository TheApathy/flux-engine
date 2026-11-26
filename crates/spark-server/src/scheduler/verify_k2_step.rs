// SPDX-License-Identifier: AGPL-3.0-only

//! K=2 verify step.

use super::*;

/// K=2 verify: [last_token, draft] → [v0, v1]. Two outcomes: ACCEPT or REJECT.
pub fn step_verify_k2(model: &dyn Model, a: &mut ActiveSeq, drafts: &[u32], num_drafts: usize) {
    let t_sync = Instant::now();
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        fail_speculative_step(a, "K2 sync_secondary", e);
        return;
    }
    let sync_us = t_sync.elapsed().as_micros();

    // EP: broadcast verify K=2 command + tokens so worker runs decode_verify_graphed in lockstep.
    let t_ep = Instant::now();
    let tokens_k2 = [a.last_token, drafts[0]];
    if let Err(e) = model.ep_broadcast_cmd(0xFFFFFFF2) {
        tracing::error!("EP broadcast verify_k2 cmd: {e:#}");
        fail_speculative_step(a, "K2 EP command", e);
        return;
    }
    for &t in &tokens_k2 {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast verify_k2 token: {e:#}");
            fail_speculative_step(a, "K2 EP token", e);
            return;
        }
    }

    let ep_us = t_ep.elapsed().as_micros();

    let mut frame = ShortVerifyFrame::capture(a, 2);
    let t_verify = Instant::now();
    let result = match model.decode_verify_graphed(&tokens_k2, &mut a.seq, 0) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("decode_verify_graphed: {e:#}");
            frame.fail(model, a, "K2 decode_verify", e);
            return;
        }
    };
    let verify_us = t_verify.elapsed().as_micros();
    a.last_token_time = Instant::now();
    let [v0_argmax, v1_argmax] = result;

    // Use argmax for speculative acceptance. Temperature-aware resampling
    // (verify_resample) requires blocking D2H copy (~0.8ms/step overhead).
    // TODO: implement GPU-side temperature sampling to avoid D2H penalty.
    let v0 = v0_argmax;
    let v1 = v1_argmax;
    let accepted = drafts[0] == v0;

    // Per-position acceptance diagnostics (ATLAS_MTP_ACCEPT_LOG=1, inert
    // otherwise). Draft slot 0 compares against verified[0]; the bonus row v1
    // is the target's own next token after the draft, not a draft slot.
    record_accept(
        if model.proposer_is_dflash() {
            "dflash"
        } else {
            "mtp"
        },
        a.seq.seq_len,
        num_drafts,
        drafts,
        &[v0],
        usize::from(accepted),
    );

    // Extract logprobs from verify logits buffer (K=2 positions) when requested.
    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        extract_verify_logprobs(model, &[v0, v1], top_logprobs)
    } else {
        Vec::new()
    };

    // EP: always broadcast accept/reject to worker (prevents deadlock on EOS).
    if let Err(e) = model.ep_broadcast_cmd(accepted as u32) {
        tracing::error!("EP broadcast verify_k2 result: {e:#}");
        frame.fail(model, a, "K2 EP result", e);
        return;
    }

    // EASD scaffolding (A.2): track baseline accept rate to decide
    // whether activating entropy-aware-spec-decode (per-step D2H of
    // verify logits + entropy threshold per arXiv:2512.23765) is
    // worth its overhead. Baseline acceptance is the prerequisite
    // signal — high accept rate means EASD has little to gain.
    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&["2", if accepted { "accept" } else { "reject" }])
        .inc();
    crate::metrics::compat_metrics::record_spec_decode(1, u64::from(accepted));
    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());

    if accepted {
        // ── ACCEPTED ──
        frame.emit(a, drafts[0], verify_lps.first().cloned());
        if !a.finished {
            frame.emit(a, v1, verify_lps.get(1).cloned());
        }
        if a.finished {
            frame.reconcile_terminal(model, a, 1, &tokens_k2);
            return;
        }
        a.last_token = v1;

        // F62 (2026-04-27): SpecMamba commit. Full accept (num_accepted=k=2):
        // copy verify scratch → canonical state.
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 2, 2) {
            tracing::error!("commit_verify_state_async (accept): {e:#}");
            frame.fail(model, a, "K2 commit accepted prefix", e);
            return;
        }
        if let Err(e) = save_hidden_for_active_proposer(model, a, v1, 1) {
            tracing::error!("save hidden for active proposer (K2 row 1): {e:#}");
            fail_speculative_step(a, "K2 save accepted hidden", e);
            return;
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 1, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
            fail_speculative_step(a, "K2 trim accepted proposer", e);
            return;
        }
        frame.settled(a, 1);
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match proposal_lifecycle::propose_and_install(
            model,
            a,
            v1,
            num_drafts,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
                fail_speculative_step(a, "K2 propose accepted prefix", e);
                return;
            }
        }
        let propose_us = t_propose.elapsed().as_micros();
        if a.seq.seq_len.is_multiple_of(50) {
            tracing::info!(
                "K2 ACCEPT: ep={ep_us}μs sync={sync_us}μs verify={verify_us}μs propose={propose_us}μs seq_len={}",
                a.seq.seq_len
            );
        }
    } else {
        // ── REJECTED ──
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();

        frame.emit(a, v0, verify_lps.first().cloned());
        if a.finished {
            frame.reconcile_terminal(model, a, 0, &tokens_k2);
            return;
        }
        a.last_token = v0;

        // F62 (2026-04-27): SpecMamba commit. K=2 reject means
        // num_accepted=1 (last_token is always accepted): copy
        // intermediate[0] → canonical. Verify scratch is discarded.
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 1, 2) {
            tracing::error!("commit_verify_state_async (reject): {e:#}");
            frame.fail(model, a, "K2 commit rejected prefix", e);
            return;
        }

        if let Err(e) = save_hidden_for_active_proposer(model, a, v0, 0) {
            tracing::error!("save hidden for active proposer (K2 row 0): {e:#}");
            fail_speculative_step(a, "K2 save rejected hidden", e);
            return;
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
            fail_speculative_step(a, "K2 trim rejected proposer", e);
            return;
        }
        frame.settled(a, 0);
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match proposal_lifecycle::propose_and_install(
            model,
            a,
            v0,
            num_drafts,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
                fail_speculative_step(a, "K2 propose rejected prefix", e);
                return;
            }
        }
        let propose_us = t_propose.elapsed().as_micros();
        let new_draft = a.pending_drafts.first().copied().unwrap_or(0);
        tracing::debug!(
            "K2 REJECT: ep={ep_us}μs sync={sync_us}μs verify={verify_us}μs propose={propose_us}μs seq_len={} last_tok={} prev_draft={} v0_verified={} new_draft={}",
            a.seq.seq_len,
            a.last_token,
            drafts[0],
            v0,
            new_draft,
        );
        if a.seq.seq_len.is_multiple_of(50) {
            tracing::info!(
                "K2 REJECT sample: ep={ep_us}μs sync={sync_us}μs verify={verify_us}μs propose={propose_us}μs seq_len={} last_tok={} prev_draft={} v0_verified={} new_draft={}",
                a.seq.seq_len,
                a.last_token,
                drafts[0],
                v0,
                new_draft,
            );
        }
    }
}

#[cfg(test)]
#[path = "verify_k2_cleanup_tests.rs"]
mod tests;
