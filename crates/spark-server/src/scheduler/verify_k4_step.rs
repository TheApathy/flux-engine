// SPDX-License-Identifier: AGPL-3.0-only

//! K=4 verify step.

use super::*;

/// K=4 verify: [last_token, draft1, draft2, draft3] → [v0, v1, v2, v3].
/// Four outcomes: accept 0, 1, 2, or 3 drafts.
pub fn step_verify_k4(model: &dyn Model, a: &mut ActiveSeq, drafts: &[u32], num_drafts: usize) {
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        fail_speculative_step(a, "K4 sync_secondary", e);
        return;
    }

    let tokens_k4 = [a.last_token, drafts[0], drafts[1], drafts[2]];

    // EP: broadcast verify K=4 command + 4 tokens.
    if let Err(e) = model.ep_broadcast_cmd(0xFFFFFFF4) {
        tracing::error!("EP broadcast verify_k4 cmd: {e:#}");
        fail_speculative_step(a, "K4 EP command", e);
        return;
    }
    for &t in &tokens_k4 {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast verify_k4 token: {e:#}");
            fail_speculative_step(a, "K4 EP token", e);
            return;
        }
    }

    let mut frame = ShortVerifyFrame::capture(a, 4);
    let t_verify = Instant::now();
    let result = match model.decode_verify_graphed_k4(&tokens_k4, &mut a.seq, 0) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("decode_verify_graphed_k4: {e:#}");
            frame.fail(model, a, "K4 decode_verify", e);
            return;
        }
    };
    let verify_us = t_verify.elapsed().as_micros();
    a.last_token_time = Instant::now();
    let [v0_argmax, v1_argmax, v2_argmax, v3_argmax] = result;

    // Use argmax for speculative acceptance (see K=2 comment).
    let v0 = v0_argmax;
    let v1 = v1_argmax;
    let v2 = v2_argmax;
    let v3 = v3_argmax;

    let num_accepted = if drafts[0] != v0 {
        0
    } else if drafts[1] != v1 {
        1
    } else if drafts[2] != v2 {
        2
    } else {
        3
    };

    // Per-position acceptance diagnostics (ATLAS_MTP_ACCEPT_LOG=1, inert
    // otherwise). Draft slots 0..=2 compare against verified[0..=2].
    record_accept(
        if model.proposer_is_dflash() {
            "dflash"
        } else {
            "mtp"
        },
        a.seq.seq_len,
        num_drafts,
        drafts,
        &[v0, v1, v2],
        num_accepted,
    );

    // Extract logprobs from verify logits buffer (K=4 positions) when requested.
    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        extract_verify_logprobs(model, &[v0, v1, v2, v3], top_logprobs)
    } else {
        Vec::new()
    };

    // EP: broadcast num_accepted to worker.
    if let Err(e) = model.ep_broadcast_cmd(num_accepted as u32) {
        tracing::error!("EP broadcast verify_k4 result: {e:#}");
        frame.fail(model, a, "K4 EP result", e);
        return;
    }

    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&[
            "4",
            match num_accepted {
                3 => "accept_all",
                0 => "reject",
                _ => "accept_partial",
            },
        ])
        .inc();
    crate::metrics::compat_metrics::record_spec_decode(3, num_accepted as u64);
    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());

    // Per-verify trace at debug — fires every 1-4 output tokens during
    // spec-decode and spams Docker logs at info level. Power-user
    // diagnostics: `RUST_LOG=spark::scheduler::verify_k4_step=debug`.
    tracing::debug!(
        "K4 verify: tokens=[{},{},{},{}] → v=[{v0},{v1},{v2},{v3}] drafts=[{},{},{}] accepted={num_accepted} seq_len={}",
        tokens_k4[0],
        tokens_k4[1],
        tokens_k4[2],
        tokens_k4[3],
        drafts[0],
        drafts[1],
        drafts[2],
        a.seq.seq_len
    );

    if num_accepted == 3 {
        frame.emit(a, drafts[0], verify_lps.first().cloned());
        if !a.finished {
            frame.emit(a, drafts[1], verify_lps.get(1).cloned());
        }
        if !a.finished {
            frame.emit(a, drafts[2], verify_lps.get(2).cloned());
        }
        if !a.finished {
            frame.emit(a, v3, verify_lps.get(3).cloned());
        }
        if a.finished {
            frame.reconcile_terminal(model, a, num_accepted, &tokens_k4);
            return;
        }
        a.last_token = v3;

        // F62/F63 (2026-04-27): SpecMamba commit. K=4 full accept.
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 4, 4) {
            tracing::error!("commit_verify_state_async (K=4 accept-4): {e:#}");
            frame.fail(model, a, "K4 commit accepted prefix", e);
            return;
        }
        if let Err(e) = save_hidden_for_active_proposer(model, a, v3, 3) {
            tracing::error!("save hidden for active proposer (K4 row 3): {e:#}");
            fail_speculative_step(a, "K4 save proposer hidden", e);
            return;
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 3, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
            fail_speculative_step(a, "K4 trim proposer state", e);
            return;
        }
        frame.settled(a, 3);
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match proposal_lifecycle::propose_and_install(
            model,
            a,
            v3,
            num_drafts,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
                fail_speculative_step(a, "K4 propose next frame", e);
                return;
            }
        }
        let propose_us = t_propose.elapsed().as_micros();
        if a.seq.seq_len.is_multiple_of(50) {
            tracing::info!(
                "K4 ACCEPT-3: verify={verify_us}μs propose={propose_us}μs seq_len={}",
                a.seq.seq_len
            );
        }
    } else if num_accepted == 2 {
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();
        frame.emit(a, drafts[0], verify_lps.first().cloned());
        if !a.finished {
            frame.emit(a, drafts[1], verify_lps.get(1).cloned());
        }
        if !a.finished {
            frame.emit(a, v2, verify_lps.get(2).cloned());
        }
        if a.finished {
            frame.reconcile_terminal(model, a, num_accepted, &tokens_k4);
            return;
        }
        a.last_token = v2;
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 3, 4) {
            tracing::error!("commit_verify_state_async (K=4 accept-3): {e:#}");
            frame.fail(model, a, "K4 commit partial prefix", e);
            return;
        }
        if let Err(e) = save_hidden_for_active_proposer(model, a, v2, 2) {
            tracing::error!("save hidden for active proposer (K4 row 2): {e:#}");
            fail_speculative_step(a, "K4 save proposer hidden", e);
            return;
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 2, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
            fail_speculative_step(a, "K4 trim proposer state", e);
            return;
        }
        frame.settled(a, 2);
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match proposal_lifecycle::propose_and_install(
            model,
            a,
            v2,
            num_drafts,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
                fail_speculative_step(a, "K4 propose next frame", e);
                return;
            }
        }
        let propose_us = t_propose.elapsed().as_micros();
        // Rate-limit to every 50 tokens (matches ACCEPT-3 above and the
        // K2 ACCEPT line in verify_k2_step.rs:119). Full per-event traces
        // available via `RUST_LOG=spark::scheduler::verify_k4_step=debug`.
        tracing::debug!(
            "K4 ACCEPT-2: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        if a.seq.seq_len.is_multiple_of(50) {
            tracing::info!(
                "K4 ACCEPT-2 sample: verify={verify_us}μs propose={propose_us}μs seq_len={}",
                a.seq.seq_len
            );
        }
    } else if num_accepted == 1 {
        a.seq.seq_len -= 2;
        a.seq.tokens.pop();
        a.seq.tokens.pop();
        frame.emit(a, drafts[0], verify_lps.first().cloned());
        if !a.finished {
            frame.emit(a, v1, verify_lps.get(1).cloned());
        }
        if a.finished {
            frame.reconcile_terminal(model, a, num_accepted, &tokens_k4);
            return;
        }
        a.last_token = v1;
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 2, 4) {
            tracing::error!("commit_verify_state_async (K=4 accept-2): {e:#}");
            frame.fail(model, a, "K4 commit partial prefix", e);
            return;
        }
        if let Err(e) = save_hidden_for_active_proposer(model, a, v1, 1) {
            tracing::error!("save hidden for active proposer (K4 row 1): {e:#}");
            fail_speculative_step(a, "K4 save proposer hidden", e);
            return;
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 1, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
            fail_speculative_step(a, "K4 trim proposer state", e);
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
                fail_speculative_step(a, "K4 propose next frame", e);
                return;
            }
        }
        let propose_us = t_propose.elapsed().as_micros();
        tracing::debug!(
            "K4 ACCEPT-1: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        if a.seq.seq_len.is_multiple_of(50) {
            tracing::info!(
                "K4 ACCEPT-1 sample: verify={verify_us}μs propose={propose_us}μs seq_len={}",
                a.seq.seq_len
            );
        }
    } else {
        a.seq.seq_len -= 3;
        a.seq.tokens.pop();
        a.seq.tokens.pop();
        a.seq.tokens.pop();
        frame.emit(a, v0, verify_lps.first().cloned());
        if a.finished {
            frame.reconcile_terminal(model, a, num_accepted, &tokens_k4);
            return;
        }
        a.last_token = v0;
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 1, 4) {
            tracing::error!("commit_verify_state_async (K=4 accept-1): {e:#}");
            frame.fail(model, a, "K4 commit rejected prefix", e);
            return;
        }
        if let Err(e) = save_hidden_for_active_proposer(model, a, v0, 0) {
            tracing::error!("save hidden for active proposer (K4 row 0): {e:#}");
            fail_speculative_step(a, "K4 save proposer hidden", e);
            return;
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
            fail_speculative_step(a, "K4 trim proposer state", e);
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
                fail_speculative_step(a, "K4 propose next frame", e);
                return;
            }
        }
        let propose_us = t_propose.elapsed().as_micros();
        tracing::debug!(
            "K4 REJECT: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        if a.seq.seq_len.is_multiple_of(50) {
            tracing::info!(
                "K4 REJECT sample: verify={verify_us}μs propose={propose_us}μs seq_len={}",
                a.seq.seq_len
            );
        }
    }
}
