// SPDX-License-Identifier: AGPL-3.0-only

//! Self-speculative + NGram speculative decoding step + grammar helpers.

use super::*;

fn abort_self_spec_verify(
    model: &dyn Model,
    a: &mut ActiveSeq,
    pre_verify_len: usize,
    k_verify: usize,
) {
    a.seq.seq_len = pre_verify_len;
    a.seq.tokens.truncate(pre_verify_len);
    if let Err(rollback_error) =
        super::verify_dflash_step::rollback_failed_verify_frame(model, &mut a.seq, k_verify)
    {
        tracing::error!("self-spec verify abort rollback failed: {rollback_error:#}");
    }
    a.finished = true;
}

/// Self-speculative step: draft via layer-skipping, verify with full model.
/// Combines bootstrap + verify in one step (no pipeline).
pub fn step_self_spec(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    num_drafts: usize,
    sync_pending: &mut bool,
    think: &ThinkSpecCtx<'_>,
) {
    let a = &mut active[0];

    // `commit_verify_state_async` writes the accepted recurrent state and its
    // next checkpoint on the secondary stream.  Do not let the next target
    // decode read that state until the paired receipt has been ordered onto
    // the default stream.
    if *sync_pending {
        if let Err(e) = model.sync_secondary() {
            tracing::error!("self-spec commit receipt sync: {e:#}");
            a.finished = true;
            return;
        }
        *sync_pending = false;
    }
    let phase_was_thinking = a.inside_thinking;

    // 1. Full-model decode to get token_0
    if let Err(e) = model.ep_broadcast_cmd(a.last_token) {
        tracing::error!("EP broadcast self-spec token: {e:#}");
        a.finished = true;
        return;
    }
    let logits = match model.decode(a.last_token, &mut a.seq, 0) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("self-spec decode error: {e:#}");
            a.finished = true;
            return;
        }
    };
    let token_0 = if phase_was_thinking {
        if !think.enabled {
            tracing::error!("self-spec cannot apply thinking policy to FP32 verify logits");
            a.finished = true;
            return;
        }
        bootstrap_thinking_token(model, a, logits, think)
    } else {
        bootstrap_content_token(model, a, logits, think)
    };
    let Some(token_0) = token_0 else {
        return;
    };
    if a.finished {
        return;
    }
    a.last_token = token_0;
    // Never draft across a thinking boundary. The committed boundary token is
    // the bonus for this step and is fed once under the new phase next tick.
    if a.inside_thinking != phase_was_thinking {
        return;
    }

    // 2. Draft phase: layer-skipping for cheap predictions.
    //
    // SPARSITY-DRAFTED SELF-SPECULATION (ATLAS_SELF_SPEC_SPARSE=1): route the
    // draft through the column-sparse FFN path (`decode_draft_sparse`) at the
    // configured keep threshold. When OFF, call the ORIGINAL `decode_draft`
    // byte-for-byte (the `use_sparse=false` branch below). The sparse draft is
    // only a proposer — the dense verify (step 5) is the lossless oracle.
    let use_sparse = spark_model::layers::self_spec_sparse_enabled();
    let sparse_thresh = spark_model::layers::self_spec_sparse_thresh();
    let seq_len_before_draft = a.seq.seq_len;
    let tokens_before_draft = a.seq.tokens.len();

    let mut draft_tokens = Vec::with_capacity(num_drafts);
    let mut draft_token = token_0;
    for _ in 0..num_drafts {
        let draft_res = if use_sparse {
            model.decode_draft_sparse(draft_token, sparse_thresh, &mut a.seq, 0)
        } else {
            model.decode_draft(draft_token, &mut a.seq, 0)
        };
        let logits = match draft_res {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("self-spec draft error: {e:#}");
                break;
            }
        };
        draft_token = match model.argmax_on_device(logits, 0) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("self-spec draft argmax error: {e:#}");
                break;
            }
        };
        draft_tokens.push(draft_token);
    }

    // 3. Rewind to pre-draft state (SSM unchanged since we skipped SSM layers)
    a.seq.seq_len = seq_len_before_draft;
    a.seq.tokens.truncate(tokens_before_draft);

    if draft_tokens.is_empty() {
        // Bootstrap already committed token_0 through the plain policy oracle.
        return;
    }

    // 4. Publish the just-completed serial state as the canonical verify
    // checkpoint.  The variable-K verifier copies canonical -> scratch and
    // records per-token intermediates, so partial accepts remain valid under
    // lazy GDN as well as the ordinary eager recurrent kernels.
    if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
        tracing::error!("self-spec start checkpoint: {e:#}");
        a.finished = true;
        return;
    }
    if let Err(e) = model.sync_secondary() {
        tracing::error!("self-spec checkpoint receipt sync: {e:#}");
        a.finished = true;
        return;
    }
    let seq_len_before_verify = a.seq.seq_len;

    // 5. Verify: run full model on [token_0, d1, ..., dK]
    let mut verify_tokens = vec![token_0];
    verify_tokens.extend_from_slice(&draft_tokens);

    let verified = match model.decode_verify_graphed_kgamma(&verify_tokens, &mut a.seq, 0) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("self-spec verify error: {e:#}");
            abort_self_spec_verify(model, a, seq_len_before_verify, verify_tokens.len());
            return;
        }
    };

    if verified.len() != draft_tokens.len() + 1 {
        tracing::error!(
            expected = draft_tokens.len() + 1,
            actual = verified.len(),
            "self-spec verify returned an invalid row count"
        );
        abort_self_spec_verify(model, a, seq_len_before_verify, verify_tokens.len());
        return;
    }

    // 6. Re-run every target row through the same request policy as serial
    // decode. This preserves sampling, penalties, masks, grammar side effects,
    // logprobs, and thinking boundaries; raw argmax is used only when the
    // shared oracle proves it equivalent.
    let started_thinking = a.inside_thinking;
    let outcome = if a.inside_thinking {
        run_dflash_thinking_accept(model, a, &draft_tokens, &verified, think)
    } else {
        run_dflash_content_accept(model, a, &draft_tokens, &verified, think)
    };
    let Some(outcome) = outcome else {
        tracing::error!("self-spec policy oracle unavailable; finishing sequence");
        abort_self_spec_verify(model, a, seq_len_before_verify, verify_tokens.len());
        return;
    };
    let num_accepted = outcome.num_accepted;
    if num_accepted > draft_tokens.len() {
        tracing::error!(
            num_accepted,
            drafts = draft_tokens.len(),
            "self-spec invalid accept count"
        );
        abort_self_spec_verify(model, a, seq_len_before_verify, verify_tokens.len());
        return;
    }

    // 7. Rollback extra verify tokens
    // tokens_added = token_0 (always kept) + accepted drafts
    let tokens_added = 1 + num_accepted;
    let expected_seq_len = seq_len_before_verify + tokens_added;

    if a.seq.seq_len < expected_seq_len || a.seq.tokens.len() < expected_seq_len {
        tracing::error!(
            seq_len = a.seq.seq_len,
            token_len = a.seq.tokens.len(),
            expected_seq_len,
            "self-spec verify state shorter than committed prefix"
        );
        abort_self_spec_verify(model, a, seq_len_before_verify, verify_tokens.len());
        return;
    }
    a.seq.seq_len = expected_seq_len;
    a.seq.tokens.truncate(expected_seq_len);
    if let Err(e) = model.commit_verify_state_async(&mut a.seq, tokens_added, verify_tokens.len()) {
        tracing::error!("self-spec commit_verify_state_async: {e:#}");
        abort_self_spec_verify(model, a, seq_len_before_verify, verify_tokens.len());
        return;
    }
    *sync_pending = true;
    if started_thinking && super::think_spec_accept::outcome_ended_thinking(a, &outcome) {
        if let Err(e) = model.sync_secondary() {
            tracing::error!("self-spec sync_secondary before SSM boundary snapshot: {e:#}");
            abort_self_spec_verify(model, a, seq_len_before_verify, verify_tokens.len());
            return;
        }
        *sync_pending = false;
        rollback::snapshot_boundary_if_ssm(a, model);
    }
    if !a.finished
        && let Some(bonus) = outcome.bonus
    {
        a.last_token = bonus;
    }
    tracing::debug!(
        accepted = num_accepted,
        drafts = draft_tokens.len(),
        committed = 1 + num_accepted + usize::from(outcome.bonus.is_some()),
        "SELF_SPEC_COMMIT_RECEIPT"
    );
}

/// Drain the accepted-state receipt before a self-spec request changes modes
/// or is retired.  A failed drain makes every active sequence non-cacheable by
/// finishing it; continuing serial decode from an unordered recurrent state
/// would silently corrupt all later tokens.
pub fn abandon_self_spec_frame(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sync_pending: &mut bool,
) -> bool {
    if !*sync_pending {
        return true;
    }
    *sync_pending = false;
    match model.sync_secondary() {
        Ok(()) => true,
        Err(e) => {
            tracing::error!("self-spec abandon sync_secondary: {e:#}");
            for a in active.iter_mut() {
                a.finished = true;
            }
            false
        }
    }
}

/// N-gram speculative step: CPU proposer + CUDA-graphed K=2 verify.
///
/// Two-phase pipeline (same as MTP but with N-gram proposer instead):
/// 1. Bootstrap: regular decode → argmax → N-gram propose → pending_drafts
/// 2. Verify: decode_verify_graphed(K=2) → accept/reject → SSM rollback
pub fn step_ngram(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    proposer: &mut NgramProposer,
    sync_pending: &mut bool,
    think: &ThinkSpecCtx<'_>,
) {
    let a = &mut active[0];

    if *sync_pending {
        if let Err(e) = model.sync_secondary() {
            tracing::error!("ngram checkpoint receipt sync: {e:#}");
            a.pending_drafts.clear();
            a.finished = true;
            return;
        }
        *sync_pending = false;
    }

    if !a.pending_drafts.is_empty() {
        // ── Phase B: Verify pending draft ──
        let drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        step_ngram_verify(model, a, &drafts, proposer, sync_pending, think);
    } else {
        // ── Phase A: Bootstrap decode + N-gram propose ──
        if let Err(e) = model.ep_broadcast_cmd(a.last_token) {
            tracing::error!("EP broadcast ngram bootstrap: {e:#}");
            a.finished = true;
            return;
        }
        let logits = match model.decode(a.last_token, &mut a.seq, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("ngram bootstrap decode error: {e:#}");
                a.finished = true;
                return;
            }
        };
        let phase_was_thinking = a.inside_thinking;
        let tok = if phase_was_thinking {
            if !think.enabled {
                tracing::error!("ngram cannot apply thinking policy to FP32 verify logits");
                a.finished = true;
                return;
            }
            bootstrap_thinking_token(model, a, logits, think)
        } else {
            bootstrap_content_token(model, a, logits, think)
        };
        let Some(tok) = tok else {
            return;
        };
        if a.finished {
            return;
        }
        a.last_token = tok;
        if a.inside_thinking != phase_was_thinking {
            return;
        }

        // `tok` is committed but intentionally not in seq.tokens until the
        // next target call. Query the logical history including that token;
        // using seq.tokens alone makes every proposal one token stale.
        if let Some(draft) =
            a.self_context
                .propose_one_after(&a.seq.tokens, tok, proposer.min_match())
        {
            // Checkpoint SSM for potential rollback during verify
            if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
                tracing::error!("ngram start_checkpoint_async: {e:#}");
                a.finished = true;
                return;
            }
            a.pending_drafts = vec![draft];
            *sync_pending = true;
        }
        // If no proposal: next iteration will be another bootstrap (regular decode)
    }
}

/// Verify a single N-gram draft via CUDA-graphed K=2 path.
pub fn step_ngram_verify(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    proposer: &mut NgramProposer,
    sync_pending: &mut bool,
    think: &ThinkSpecCtx<'_>,
) {
    if drafts.len() != 1 {
        tracing::error!(
            actual = drafts.len(),
            "ngram verify requires exactly one draft"
        );
        a.finished = true;
        return;
    }

    // EP: broadcast verify K=2 command + tokens
    let tokens_k2 = [a.last_token, drafts[0]];
    if let Err(e) = model.ep_broadcast_cmd(0xFFFFFFF2) {
        tracing::error!("EP broadcast ngram verify cmd: {e:#}");
        a.finished = true;
        return;
    }
    for &t in &tokens_k2 {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast ngram verify token: {e:#}");
            a.finished = true;
            return;
        }
    }

    let t_verify = Instant::now();
    let result = match model.decode_verify_graphed(&tokens_k2, &mut a.seq, 0) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("ngram decode_verify_graphed: {e:#}");
            a.finished = true;
            return;
        }
    };
    let verify_us = t_verify.elapsed().as_micros();
    a.last_token_time = Instant::now();
    let [v0, v1] = result;

    // EP: broadcast accept/reject to worker
    let started_thinking = a.inside_thinking;
    let outcome = if a.inside_thinking {
        if !think.enabled {
            tracing::error!("ngram verify cannot apply thinking policy to FP32 logits");
            a.finished = true;
            None
        } else {
            run_dflash_thinking_accept(model, a, drafts, &[v0, v1], think)
        }
    } else {
        run_dflash_content_accept(model, a, drafts, &[v0, v1], think)
    };
    let Some(outcome) = outcome else {
        a.finished = true;
        return;
    };
    if outcome.num_accepted > 1 {
        tracing::error!(
            accepted = outcome.num_accepted,
            "ngram invalid accept count"
        );
        a.finished = true;
        return;
    }
    let accepted = outcome.num_accepted == 1;
    if let Err(e) = model.ep_broadcast_cmd(accepted as u32) {
        tracing::error!("EP broadcast ngram verify result: {e:#}");
        a.finished = true;
        return;
    }

    // Verify fed [last_token, draft]. Retain the draft input iff the policy
    // oracle accepted it; the committed bonus remains unfed in last_token.
    if !accepted {
        if a.seq.seq_len == 0 || a.seq.tokens.is_empty() {
            tracing::error!("ngram verify state underflow during rollback");
            a.finished = true;
            return;
        }
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();
    }
    if accepted {
        proposer.record_accept();
    } else {
        proposer.record_reject();
    }

    let next_draft = if !a.finished {
        outcome.bonus.and_then(|bonus| {
            a.last_token = bonus;
            a.self_context
                .propose_one_after(&a.seq.tokens, bonus, proposer.min_match())
        })
    } else {
        None
    };

    if accepted {
        if let Some(draft) = next_draft {
            if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
                tracing::error!("ngram accept checkpoint: {e:#}");
                a.finished = true;
                return;
            }
            a.pending_drafts = vec![draft];
            *sync_pending = true;
        }
    } else {
        // Restore the live recurrent state even when no next draft exists.
        // The following bootstrap must not race this operation.
        if let Err(e) = model.start_rollback_and_checkpoint_async(&mut a.seq, 1) {
            tracing::error!("ngram rollback: {e:#}");
            a.finished = true;
            return;
        }
        if let Some(draft) = next_draft {
            a.pending_drafts = vec![draft];
        }
        *sync_pending = true;
    }
    if started_thinking && super::think_spec_accept::outcome_ended_thinking(a, &outcome) {
        if *sync_pending {
            if let Err(e) = model.sync_secondary() {
                tracing::error!("ngram sync_secondary before SSM boundary snapshot: {e:#}");
                a.pending_drafts.clear();
                a.finished = true;
                return;
            }
            *sync_pending = false;
        }
        rollback::snapshot_boundary_if_ssm(a, model);
    }

    if proposer.len() == 1 || proposer.len().is_multiple_of(50) {
        tracing::info!(
            accepted,
            verify_us,
            accepts = proposer.accepts,
            rejects = proposer.rejects,
            seq_len = a.seq.seq_len,
            "NGRAM_SPEC_COMMIT_RECEIPT"
        );
    }
}

/// Abandon an n-gram frame before serial fallback or sequence retirement.
/// Pending drafts and their asynchronous checkpoint/rollback form one receipt:
/// neither may outlive the mode transition.
pub fn abandon_ngram_frame(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sync_pending: &mut bool,
) -> bool {
    for a in active.iter_mut() {
        a.pending_drafts.clear();
    }
    if !*sync_pending {
        return true;
    }
    *sync_pending = false;
    match model.sync_secondary() {
        Ok(()) => true,
        Err(e) => {
            tracing::error!("ngram abandon sync_secondary: {e:#}");
            for a in active.iter_mut() {
                a.finished = true;
            }
            false
        }
    }
}

/// Fill the XGrammar bitmask for the current matcher position and clone it
/// into an owned `Vec<i32>` the caller can pass into MTP draft sampling.
///
/// Returns `None` when grammar is inactive, the sequence is currently inside
/// a `<think>` span (matcher is paused), the grammar has already terminated,
/// or `fill_bitmask` reported no constraint. In all those cases MTP should
/// fall back to its unconstrained GPU-argmax path.
///
/// The owned copy is small (~ceil(vocab/32)*4 bytes, ~32KB for 100k vocab)
/// and is necessary because the matcher is borrowed mutably by the scheduler
/// between `fill_bitmask` and the subsequent `accept_token` calls inside
/// `emit_token`, while the MTP propose call borrows the model immutably —
/// cloning sidesteps the lifetime overlap.
pub fn mtp_grammar_mask_for(a: &mut ActiveSeq) -> Option<Vec<i32>> {
    if a.inside_thinking {
        return None;
    }
    let gs = a.grammar_state.as_mut()?;
    if gs.is_terminated() {
        return None;
    }
    if !gs.fill_bitmask() {
        return None;
    }
    Some(gs.bitmask_data().to_vec())
}

/// Grammar-constraint handling for the DFlash proposer
/// (`ATLAS_DFLASH_GRAMMAR_MODE`, read once at first use).
///
/// Unlike MTP — which threads the XGrammar bitmask through
/// `run_mtp_propose_multi` into a masked CPU argmax — DFlash's γ-block
/// draft argmax AND the K=γ verify argmax are GPU-side and grammar-blind,
/// so an unhandled grammar (`tools` / `response_format`) was silently
/// ignored on the DFlash path. Modes:
///   - `off`:    legacy behavior (grammar ignored by DFlash — unsafe).
///   - `gate`:   (default) skip DFlash drafting while the grammar
///               constrains the next token; the sequence falls back to the
///               bootstrap single-token decode, which samples through
///               `sample_token_with_grammar`.
///   - `verify`: keep drafting; enforce the bitmask target-side in
///               `step_verify_dflash` via per-position masked argmax over
///               the verify logits (DDTree topologies still gate — the
///               tree-frame accept walk has no masked equivalent).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DflashGrammarMode {
    Off,
    Gate,
    Verify,
}

pub fn dflash_grammar_mode() -> DflashGrammarMode {
    static MODE: std::sync::OnceLock<DflashGrammarMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("ATLAS_DFLASH_GRAMMAR_MODE").ok().as_deref() {
            Some("off") => DflashGrammarMode::Off,
            Some("verify") => DflashGrammarMode::Verify,
            Some("gate") | None => DflashGrammarMode::Gate,
            Some(other) => {
                tracing::warn!(
                    "ATLAS_DFLASH_GRAMMAR_MODE={other:?} unrecognized (off|gate|verify); \
                     defaulting to \"gate\""
                );
                DflashGrammarMode::Gate
            }
        },
    )
}

/// Stage-1 grammar gate: true when the DFlash proposer must NOT draft for
/// `a` this step, i.e. the grammar actively constrains the next token and
/// the configured mode keeps DFlash out of grammar territory. The caller
/// skips `run_mtp_propose_multi`, leaving `pending_drafts` empty so the
/// sequence runs the grammar-enforced bootstrap decode instead.
pub fn dflash_grammar_skip_propose(model: &dyn Model, a: &mut ActiveSeq) -> bool {
    if !model.proposer_is_dflash() || a.grammar_state.is_none() {
        return false;
    }
    let gate = match dflash_grammar_mode() {
        DflashGrammarMode::Off => false,
        DflashGrammarMode::Gate => true,
        // verify mode keeps drafting (mask enforced in step_verify_dflash)
        // except under DDTree, whose tree-frame accept walk is unmasked.
        DflashGrammarMode::Verify => {
            std::env::var("ATLAS_DFLASH_METHOD").ok().as_deref() == Some("ddtree")
        }
    };
    if !gate || a.inside_thinking {
        return false;
    }
    // Same activation test as `mtp_grammar_mask_for`, minus the owned copy.
    let gs = a.grammar_state.as_mut().expect("checked above");
    !gs.is_terminated() && gs.fill_bitmask()
}

/// Truncate a draft list at the first token the grammar would
/// reject *if it were the next emitted token at that draft position*.
///
/// Required for K=3+ MTP paths where `run_mtp_propose_multi` uses a
/// SINGLE bitmask snapshot (taken at the start of propose) for all N
/// drafts. The mask correctly constrains `drafts[0]` but does not
/// reflect the post-`drafts[0]` grammar state — so `drafts[1]` may
/// cross a structural boundary (e.g. `drafts[0] = </function>`
/// closing a tool body, then `drafts[1] = <parameter=` which is
/// invalid in the outer free-text grammar state).
///
/// Without this guard, the spec verifier accepts the cross-boundary
/// span (the model's actual sample matches whatever the in-tool
/// distribution happened to produce), `emit_token` advances the
/// grammar past `</function>`, and the next `accept_token` for
/// `drafts[1]` returns false silently — the token is already in
/// `output_tokens`, but the grammar is desync'd from the output
/// stream. Subsequent bitmasks are wrong.
///
/// Reference: arXiv:2512.15834 ("Speculative Tool Calls"). The
/// canonical fix is to re-run the grammar mask from a fresh outer
/// state for each draft; we approximate cheaply by simulating
/// `accept_token` per draft and truncating at the first rejection,
/// rolling the state back when done. The verifier then accepts at
/// most the validated prefix.
///
/// Returns the number of drafts that pass grammar validation.
/// Mutates `gs` transiently but restores it via `rollback`. K=2
/// (num_drafts=1) callers can skip this — a single draft uses its
/// own up-to-date mask.
pub fn truncate_drafts_at_grammar_boundary(gs: &mut GrammarState, drafts: &[u32]) -> usize {
    if drafts.len() < 2 || gs.is_terminated() {
        return drafts.len();
    }
    let history_before = gs.num_history_steps();
    let mut accepted = 0usize;
    for &tok in drafts {
        if !gs.accept_token(tok) {
            break;
        }
        accepted += 1;
    }
    // `accept_token` returns true for tokens after grammar termination so the
    // EOS handler, rather than this speculative boundary check, owns request
    // termination. Those trailing tokens add no matcher history. Rolling back
    // `accepted` therefore panics whenever an early stop token is followed by
    // more drafts (for example, 12 logical accepts but one recorded step).
    let recorded = gs.num_history_steps() - history_before;
    if recorded > 0 {
        gs.rollback(recorded);
    }
    if accepted < drafts.len() {
        tracing::warn!(
            kept = accepted,
            dropped = drafts.len() - accepted,
            "spec-decode boundary: truncated drafts crossing grammar transition"
        );
    }
    accepted
}

#[cfg(test)]
mod grammar_boundary_tests {
    use super::truncate_drafts_at_grammar_boundary;
    use crate::grammar::{GrammarEngine, GrammarState};

    fn test_vocab() -> Vec<String> {
        let mut vocab = (0u8..128)
            .map(|byte| String::from(byte as char))
            .collect::<Vec<_>>();
        vocab.push("<tool_call>".to_string());
        vocab.push("</tool_call>".to_string());
        vocab.push("<eos>".to_string());
        vocab
    }

    #[test]
    fn rollback_uses_recorded_history_when_stop_precedes_draft_tail() {
        let vocab = test_vocab();
        let mut engine = GrammarEngine::new(&vocab, &[130]).unwrap();
        let compiled = engine.compile_json_grammar().unwrap();
        let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

        assert!(state.accept_token(b'{' as u32));
        assert!(state.accept_token(b'}' as u32));
        let history_before = state.num_history_steps();

        // Reproduce the production crash shape: EOS is the only matcher step,
        // while the remaining eleven speculative tokens are logical no-ops.
        let mut drafts = vec![130];
        drafts.extend(std::iter::repeat_n(b'x' as u32, 11));
        assert_eq!(truncate_drafts_at_grammar_boundary(&mut state, &drafts), 12);
        assert_eq!(state.num_history_steps(), history_before);
        assert!(!state.is_terminated());
        assert!(state.accept_token(130));
    }
}
