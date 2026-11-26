// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-sequence batched DFlash verify scheduler step.
//!
//! Gathers `c` active sequences' K=γ+1 draft windows into ONE batched verify
//! forward (`Model::decode_verify_dflash_batched`) whose FFN GEMMs read the
//! ~14 GB of NVFP4 FFN weights ONCE for all `c×K` rows — instead of the
//! per-sequence loop where each sequence runs its own full-weight-sweep
//! verify. The per-sequence accept / emit / commit / propose logic is
//! IDENTICAL to the flat-chain path of `step_verify_dflash`, so output is
//! lossless and byte-identical to single-stream per sequence.
//!
//! ## Eligibility (all must hold for every batched sequence)
//! - `pending_drafts.len() >= 4` (DFlash γ-block) and equal K across the batch;
//! - no active grammar (`grammar_state` None or terminated);
//! - not inside `<think>` (no thinking-accept filter in the batched path);
//! - no tree payload (`pending_tree_payload` None) and no tree/portfolio draft
//!   method active (flat chain only);
//! - not finished.
//!
//! Sequences that don't qualify fall through to the per-seq path in step_mtp.
//! Gated by `ATLAS_DFLASH_BATCHED_VERIFY=1` (default OFF).

use super::*;

/// Whether the cross-seq batched DFlash verify is enabled. Read once.
pub(super) fn dflash_batched_verify_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DFLASH_BATCHED_VERIFY").ok().as_deref() == Some("1"))
}

/// Whether per-step batched DFlash verify timing logs are enabled. Read once.
fn dflash_batched_verify_log_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_BATCHED_VERIFY_LOG")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// Returns `true` when every index in `idxs` is in range `< len` and distinct.
fn batch_indices_valid(idxs: &[usize], len: usize) -> bool {
    idxs.iter()
        .enumerate()
        .all(|(i, &idx)| idx < len && !idxs[..i].contains(&idx))
}

/// Run the batched DFlash verify over `batched_idxs` (indices into `active`),
/// then finalize each sequence per-seq. Caller guarantees every listed
/// sequence satisfies the eligibility rules above and shares the same K.
pub(super) fn step_verify_dflash_batched(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    batched_idxs: &[usize],
    num_drafts: usize,
) {
    if !batch_indices_valid(batched_idxs, active.len()) {
        tracing::error!(
            "step_verify_dflash_batched: batch indices are not distinct and in range; skipping"
        );
        return;
    }
    if batched_idxs
        .iter()
        .any(|&idx| active[idx].pending_dflash_sampled_proposal.is_some())
    {
        tracing::error!(
            "sampled DFlash q frame reached the flat batched verifier; dropping the batch"
        );
        for &idx in batched_idxs {
            super::proposal_lifecycle::clear(
                &mut active[idx].pending_drafts,
                &mut active[idx].pending_tree_payload,
                &mut active[idx].pending_dflash_sampled_proposal,
            );
        }
        return;
    }
    let sync_result = super::verify_dflash_step::dflash_capacity::preflight_batch_then(
        model.proposer_is_dflash(),
        || model.dflash_verify_capacity_k(),
        num_drafts,
        batched_idxs.iter().map(|&idx| {
            (
                active[idx].pending_drafts.len(),
                active[idx].pending_tree_payload.as_ref(),
            )
        }),
        || model.sync_secondary(),
    );
    let sync_result = match sync_result {
        Ok(result) => result,
        Err(error) => {
            tracing::error!(
                "DFlash batched pre-verify capacity guard rejected the batch: {error:#}"
            );
            for &idx in batched_idxs {
                super::proposal_lifecycle::clear(
                    &mut active[idx].pending_drafts,
                    &mut active[idx].pending_tree_payload,
                    &mut active[idx].pending_dflash_sampled_proposal,
                );
            }
            return;
        }
    };
    if let Err(e) = sync_result {
        tracing::error!("sync_secondary (batched dflash): {e:#}");
        for &idx in batched_idxs {
            fail_speculative_step(&mut active[idx], "batched verify sync_secondary", &e);
        }
        return;
    }

    // Build per-seq token windows [last_token, ...drafts]. Record each seq's
    // drafts (owned) so the accept loop below can compare after the forward.
    let mut tokens_per_seq: Vec<Vec<u32>> = Vec::with_capacity(batched_idxs.len());
    let mut drafts_per_seq: Vec<Vec<u32>> = Vec::with_capacity(batched_idxs.len());
    for &idx in batched_idxs {
        let a = &mut active[idx];
        let drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        let mut toks = Vec::with_capacity(drafts.len() + 1);
        toks.push(a.last_token);
        toks.extend_from_slice(&drafts);
        tokens_per_seq.push(toks);
        drafts_per_seq.push(drafts);
    }

    // ── ONE batched verify forward: FFN weights read ONCE for all c seqs ──
    let t_verify = Instant::now();
    let mut verified_per_seq = {
        let mut seq_refs =
            match super::verify_qwen4_batched_step::batch_sequences_mut(active, batched_idxs) {
                Ok(refs) => refs,
                Err(e) => {
                    tracing::error!("batch_sequences_mut (batched dflash): {e:#}");
                    return;
                }
            };
        match model.decode_verify_dflash_batched(&tokens_per_seq, &mut seq_refs, 0) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("decode_verify_dflash_batched: {e:#}");
                for &idx in batched_idxs {
                    fail_speculative_step(&mut active[idx], "decode_verify_dflash_batched", &e);
                }
                return;
            }
        }
    };
    let t_verify_us = t_verify.elapsed().as_micros();

    // Validate the complete result before any request can emit or index a row.
    if verified_per_seq.len() != tokens_per_seq.len()
        || verified_per_seq
            .iter()
            .zip(&tokens_per_seq)
            .any(|(verified, tokens)| verified.len() != tokens.len())
    {
        let error = format!(
            "expected target row counts {:?}, received {:?}",
            tokens_per_seq.iter().map(Vec::len).collect::<Vec<_>>(),
            verified_per_seq.iter().map(Vec::len).collect::<Vec<_>>(),
        );
        tracing::error!("decode_verify_dflash_batched: {error}");
        for &idx in batched_idxs {
            fail_speculative_step(&mut active[idx], "batched verify target row count", &error);
        }
        return;
    }

    if dflash_batched_verify_log_enabled() {
        tracing::info!(
            "DFLASH BATCHED verify: c={} K={} forward={t_verify_us}μs",
            batched_idxs.len(),
            tokens_per_seq.first().map(|t| t.len()).unwrap_or(0),
        );
    }

    // ── Phase 1: per-seq accept / emit / rollback / commit / hidden-retire ──
    // Retire all surviving sequences' captured target-layer hiddens from
    // `dflash_hidden_save` BEFORE proposing so `dstate.ctx_len == seq.seq_len`
    // and Phase 2 can batch the entire cohort into one `propose_batch` pass.
    let mut propose_idxs: Vec<usize> = Vec::with_capacity(batched_idxs.len());
    for (batch_pos, &idx) in batched_idxs.iter().enumerate() {
        let a = &mut active[idx];
        let drafts = &drafts_per_seq[batch_pos];
        let verify_tokens = &tokens_per_seq[batch_pos];
        let verified = &mut verified_per_seq[batch_pos];
        let row_base = batch_pos * verify_tokens.len();
        if a.think_ended {
            super::verify_dflash_step::dflash_think_mask_verified_at_row(
                model, a, row_base, verified,
            );
        }
        if finalize_flat_dflash_verify_commit(model, a, row_base, drafts, verify_tokens, verified) {
            propose_idxs.push(idx);
        }
    }

    // ── Phase 2: batched DFlash2 propose across all surviving sequences ──
    if propose_idxs.len() >= 2 && num_drafts > 0 {
        let tokens: Vec<u32> = propose_idxs
            .iter()
            .map(|&idx| active[idx].last_token)
            .collect();
        let positions: Vec<usize> = propose_idxs
            .iter()
            .map(|&idx| active[idx].seq.seq_len)
            .collect();
        let batched_res = super::verify_qwen4_batched_step::batch_sequences_mut(
            active,
            &propose_idxs,
        )
        .and_then(|mut seq_refs| {
            model.run_mtp_propose_multi_batched(&tokens, &positions, num_drafts, &mut seq_refs, 0)
        });
        match batched_res {
            Ok(all_drafts) if all_drafts.len() == propose_idxs.len() => {
                for (&idx, drafts) in propose_idxs.iter().zip(all_drafts) {
                    let a = &mut active[idx];
                    a.pending_tree_payload = None;
                    a.pending_dflash_sampled_proposal = None;
                    if let Err(e) = proposal_lifecycle::install_collected(model, a, Ok(drafts)) {
                        tracing::error!("install proposal (batched dflash): {e:#}");
                    }
                }
                return;
            }
            Ok(all_drafts) => {
                tracing::warn!(
                    expected = propose_idxs.len(),
                    got = all_drafts.len(),
                    "run_mtp_propose_multi_batched (batched dflash) count mismatch; falling back per-seq"
                );
            }
            Err(e) => {
                tracing::warn!(
                    "run_mtp_propose_multi_batched (batched dflash) failed; falling back per-seq: {e:#}"
                );
            }
        }
    }

    for &idx in &propose_idxs {
        let a = &mut active[idx];
        let token = a.last_token;
        if let Err(e) = proposal_lifecycle::propose_and_install(model, a, token, num_drafts, None) {
            tracing::error!("run_mtp_propose_multi (batched dflash): {e:#}");
        }
    }
}

/// Flat-chain accept + emit + commit + hidden-retire for one sequence,
/// mirroring the non-tree / non-grammar / non-thinking path of
/// `step_verify_dflash` exactly. Returns `true` when the sequence remains live
/// and ready for draft proposal.
fn finalize_flat_dflash_verify_commit(
    model: &dyn Model,
    a: &mut ActiveSeq,
    row_base: usize,
    drafts: &[u32],
    verify_tokens: &[u32],
    verified: &[u32],
) -> bool {
    a.last_token_time = Instant::now();

    // Accept-prefix: drafts[i] accepted iff drafts[i] == verified[i]. Pure
    // helper (unit-tested) so the exact-match contract matches the flat path of
    // step_verify_dflash bit-for-bit.
    let num_accepted = flat_accept_prefix(drafts, verified);

    // Roll back the over-extended seq_len / seq.tokens to
    // pre_verify_len + num_accepted + 1 (accepted drafts + bonus slot).
    let k_tokens = drafts.len() + 1;
    let Some(pre_verify_len) = a.seq.seq_len.checked_sub(k_tokens) else {
        tracing::error!(
            seq_len = a.seq.seq_len,
            k_tokens,
            "batched terminal verify state underflow"
        );
        let error = format!(
            "sequence length {} is shorter than K={k_tokens}",
            a.seq.seq_len
        );
        fail_speculative_step(a, "batched verify state extent", error);
        return false;
    };
    let (target_seq_len, to_drop) = flat_rollback_plan(a.seq.seq_len, drafts.len(), num_accepted);
    if to_drop > 0 {
        a.seq.seq_len = target_seq_len;
        let pop_n = to_drop.min(a.seq.tokens.len());
        for _ in 0..pop_n {
            a.seq.tokens.pop();
        }
    }

    // Emit accepted drafts.
    let mut consumed_rows = 0usize;
    let emit_take = num_accepted.min(drafts.len());
    for &tok in &drafts[..emit_take] {
        if emit_token_for_verify(a, tok, None) {
            consumed_rows += 1;
        }
        if a.finished {
            let _lane = match spark_model::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(
                a.seq.slot_idx,
            ) {
                Ok(guard) => guard,
                Err(e) => {
                    tracing::error!("terminal verify-state lane (batched accepted draft): {e:#}");
                    fail_speculative_step(a, "terminal batched accepted-draft verify lane", &e);
                    return false;
                }
            };
            if let Err(e) = super::verify_dflash_step::reconcile_terminal_verify_state(
                model,
                a,
                pre_verify_len,
                k_tokens,
                num_accepted,
                consumed_rows,
                verify_tokens,
                false,
                None,
            ) {
                tracing::error!("terminal verify-state commit (batched accepted draft): {e:#}");
                fail_speculative_step(a, "terminal batched accepted-draft verify commit", &e);
            }
            return false;
        }
    }

    // Bonus token = verified[num_accepted].
    if let Some(&bonus) = verified.get(num_accepted) {
        if emit_token_for_verify(a, bonus, None) {
            consumed_rows += 1;
        }
        if a.finished {
            let _lane = match spark_model::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(
                a.seq.slot_idx,
            ) {
                Ok(guard) => guard,
                Err(e) => {
                    tracing::error!("terminal verify-state lane (batched bonus): {e:#}");
                    fail_speculative_step(a, "terminal batched bonus verify lane", &e);
                    return false;
                }
            };
            if let Err(e) = super::verify_dflash_step::reconcile_terminal_verify_state(
                model,
                a,
                pre_verify_len,
                k_tokens,
                num_accepted,
                consumed_rows,
                verify_tokens,
                false,
                None,
            ) {
                tracing::error!("terminal verify-state commit (batched bonus): {e:#}");
                fail_speculative_step(a, "terminal batched bonus verify commit", &e);
            }
            return false;
        }
        a.last_token = bonus;
    }

    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&[
            "dflash",
            if num_accepted == drafts.len() {
                "accept_all"
            } else {
                "accept_partial"
            },
        ])
        .inc();
    crate::metrics::compat_metrics::record_spec_decode(drafts.len() as u64, num_accepted as u64);
    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());

    // SSM commit / rollback (per-seq, bound to this slot's gdn_lazy_retain lane).
    let total_accepted = num_accepted + 1; // bonus always accepted
    let _lane =
        match spark_model::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(a.seq.slot_idx) {
            Ok(guard) => guard,
            Err(e) => {
                tracing::error!("commit_verify_state_async lane (batched dflash): {e:#}");
                fail_speculative_step(a, "batched verify commit lane", &e);
                return false;
            }
        };
    if let Err(e) = model.commit_verify_state_async(&mut a.seq, total_accepted, k_tokens) {
        tracing::error!("commit_verify_state_async (batched dflash): {e:#}");
        fail_speculative_step(a, "batched verify commit", &e);
        return false;
    }

    // Retire this sequence's accepted target-layer hidden captures from the
    // shared batched `dflash_hidden_save` frame into `dstate.ctx_hidden_acc`.
    if model.proposer_is_dflash()
        && let Err(e) =
            model.retire_dflash_flat_batch_captures(row_base, num_accepted, &mut a.seq, 0)
    {
        tracing::debug!("retire_dflash_flat_batch_captures (batched dflash): {e:#}");
    }

    // Save the bonus token's hidden for the active proposer. Cross-sequence
    // batching is DFlash-only; the shared boundary keeps the pairing explicit.
    let bonus = a.last_token;
    if let Err(e) = save_hidden_for_active_proposer(model, a, bonus, num_accepted) {
        tracing::error!("save hidden for active proposer (batched verify): {e:#}");
        return false;
    }
    if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        tracing::error!("trim_proposer_state (batched dflash): {e:#}");
    }

    true
}

/// Compatibility helper for single-sequence commit + propose tests.
#[cfg(test)]
fn finalize_flat_dflash_verify(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    verify_tokens: &[u32],
    verified: &[u32],
    num_drafts: usize,
) {
    if finalize_flat_dflash_verify_commit(model, a, 0, drafts, verify_tokens, verified) {
        let token = a.last_token;
        if let Err(e) = proposal_lifecycle::propose_and_install(model, a, token, num_drafts, None) {
            tracing::error!("run_mtp_propose_multi (batched dflash): {e:#}");
        }
    }
}

/// Pure flat-chain accept-prefix count: the length of the longest prefix where
/// `drafts[i] == verified[i]`. `verified` is `[target@last_token, target@draft0,
/// …]` (length `drafts.len()+1` in the well-formed case); a draft `i` can only
/// be accepted if its correcting/bonus slot `verified[i+1]` exists (matching the
/// `i + 1 >= verified.len()` guard in `step_verify_dflash`'s flat path).
///
/// This is the ONLY acceptance rule the batched flat path applies — grammar /
/// thinking / relax / typical / tree acceptance are all excluded by the batched
/// eligibility gate (see `mtp_step.rs`), so per-seq output stays byte-identical
/// to the single-stream flat verify.
fn flat_accept_prefix(drafts: &[u32], verified: &[u32]) -> usize {
    let mut n = 0usize;
    for i in 0..drafts.len() {
        if i + 1 >= verified.len() {
            break;
        }
        if drafts[i] == verified[i] {
            n += 1;
        } else {
            break;
        }
    }
    n
}

/// Post-verify rollback target: the batched forward advanced `seq_len` by
/// `k_tokens = drafts.len()+1`; the accepted prefix keeps
/// `pre_verify_len + num_accepted + 1` (accepted drafts + the always-accepted
/// bonus slot). Returns `(target_seq_len, num_tokens_to_pop)`.
fn flat_rollback_plan(
    current_seq_len: usize,
    drafts_len: usize,
    num_accepted: usize,
) -> (usize, usize) {
    let k_tokens = drafts_len + 1;
    let pre_verify_len = current_seq_len.saturating_sub(k_tokens);
    let target_seq_len = pre_verify_len + num_accepted + 1;
    let to_drop = current_seq_len.saturating_sub(target_seq_len);
    (target_seq_len, to_drop)
}

#[cfg(test)]
mod tests {
    use super::{flat_accept_prefix, flat_rollback_plan};

    #[test]
    fn batched_hidden_row_then_terminal_keeps_classified_prefix() {
        use crate::scheduler::ResponseSink;
        use crate::scheduler::test_fixtures::{active_seq, model};
        for suppressed_eos in [false, true] {
            let model = model::recording();
            *model.verify_failure.lock().unwrap() = Some("success");
            let mut a = active_seq(ResponseSink::Blocking(None), 3);
            let before = a.seq.seq_len;
            let tokens = [31, 41, 42, 43, 44];
            a.seq.tokens = vec![7; before];
            a.seq.tokens.extend_from_slice(&tokens);
            a.seq.seq_len += tokens.len();
            if suppressed_eos {
                a.eos_tokens = vec![41];
                a.min_tokens = 100;
                a.remaining = 2;
            } else {
                a.think_end_token = Some(41);
                a.remaining = 1;
            }
            super::finalize_flat_dflash_verify(
                &model,
                &mut a,
                &tokens[1..],
                &tokens,
                &[41, 42, 99, 99, 99],
                4,
            );
            assert!(a.finished);
            assert_eq!(a.output_tokens, [31, 42]);
            assert_eq!(*model.verify_commits.lock().unwrap(), [(before + 2, 2, 5)]);
            assert!(model.verify_aborts.lock().unwrap().is_empty());
        }
    }

    // ── flat_accept_prefix: exact-match prefix, matches step_verify_dflash ──

    #[test]
    fn accept_prefix_full_accept() {
        // All drafts match; verified has the bonus slot at the tail.
        let drafts = [10, 11, 12, 13];
        let verified = [10, 11, 12, 13, 14];
        assert_eq!(flat_accept_prefix(&drafts, &verified), 4);
    }

    #[test]
    fn accept_prefix_first_mismatch_terminates() {
        let drafts = [10, 11, 99, 13];
        let verified = [10, 11, 12, 13, 14];
        // drafts[2] (99) != verified[2] (12) → prefix stops at 2.
        assert_eq!(flat_accept_prefix(&drafts, &verified), 2);
    }

    #[test]
    fn accept_prefix_zero_when_first_diverges() {
        let drafts = [7, 11, 12, 13];
        let verified = [8, 11, 12, 13, 14];
        assert_eq!(flat_accept_prefix(&drafts, &verified), 0);
    }

    #[test]
    fn accept_prefix_capped_by_verified_bonus_slot() {
        // verified is SHORT (no bonus slot for the last draft): the guard
        // `i + 1 >= verified.len()` caps acceptance so the bonus always exists.
        let drafts = [10, 11, 12, 13];
        let verified = [10, 11, 12]; // len 3 → i can accept only 0,1 (i+1<3)
        assert_eq!(flat_accept_prefix(&drafts, &verified), 2);
    }

    #[test]
    fn accept_prefix_empty_drafts() {
        assert_eq!(flat_accept_prefix(&[], &[42]), 0);
    }

    // ── flat_rollback_plan: seq_len over-extension unwind ──

    #[test]
    fn rollback_full_accept_keeps_all_plus_bonus() {
        // pre_verify_len=100, K=5 → seq_len advanced to 105. Full accept (4):
        // target = 100 + 4 + 1 = 105 → nothing to drop.
        let (target, drop) = flat_rollback_plan(105, 4, 4);
        assert_eq!(target, 105);
        assert_eq!(drop, 0);
    }

    #[test]
    fn rollback_partial_accept_drops_rejected_tail() {
        // pre_verify_len=100, K=5, seq_len=105, accepted 2:
        // target = 100 + 2 + 1 = 103 → drop 2 (the rejected drafts 2,3).
        let (target, drop) = flat_rollback_plan(105, 4, 2);
        assert_eq!(target, 103);
        assert_eq!(drop, 2);
    }

    #[test]
    fn rollback_zero_accept_keeps_only_bonus() {
        // seq_len=105, accepted 0: target = 100 + 0 + 1 = 101 → drop 4.
        let (target, drop) = flat_rollback_plan(105, 4, 0);
        assert_eq!(target, 101);
        assert_eq!(drop, 4);
    }

    #[test]
    fn rollback_mixed_max_tokens_shapes() {
        // Distinct K per seq is fine — the plan is per-seq (mid-batch
        // compaction case): a K=3 seq at seq_len=53 with 1 accepted.
        let (target, drop) = flat_rollback_plan(53, 2, 1);
        assert_eq!(target, 53 - 3 + 1 + 1); // pre=50, +1 accepted +1 bonus =52
        assert_eq!(drop, 1);
    }

    #[test]
    fn batched_terminal_returns_reconcile_target_state() {
        let source = include_str!("verify_dflash_batched_step.rs");
        for label in [
            "terminal verify-state commit (batched accepted draft)",
            "terminal verify-state commit (batched bonus)",
        ] {
            let error_log = source.find(label).expect("batched terminal branch exists");
            let branch = &source[..error_log];
            assert!(
                branch.rfind("reconcile_terminal_verify_state(").is_some(),
                "{label} must reconcile before returning"
            );
        }
    }

    #[test]
    fn step_verify_dflash_batched_rejects_duplicate_and_oob_indices() {
        use crate::scheduler::ResponseSink;
        use crate::scheduler::test_fixtures::{active_seq, model};

        let model = model::recording();
        let mut active = vec![active_seq(ResponseSink::Blocking(None), 4)];
        active[0].pending_drafts = vec![10, 11, 12, 13];

        super::step_verify_dflash_batched(&model, &mut active, &[0, 0], 4);
        assert_eq!(active[0].pending_drafts, [10, 11, 12, 13]);

        super::step_verify_dflash_batched(&model, &mut active, &[0, 1], 4);
        assert_eq!(active[0].pending_drafts, [10, 11, 12, 13]);
    }

    #[test]
    fn batched_commit_sites_enter_per_slot_verify_lane() {
        let source = include_str!("verify_dflash_batched_step.rs");
        let fn_start = source
            .find("fn finalize_flat_dflash_verify_commit(")
            .expect("commit helper exists");
        let fn_end = source[fn_start..]
            .find("fn finalize_flat_dflash_verify(")
            .expect("test helper follows commit helper");
        let body = &source[fn_start..fn_start + fn_end];
        let lane_entries = body.matches("LaneGuard::enter_for_slot(").count();
        assert_eq!(
            lane_entries, 3,
            "all 3 batched commit sites (accepted-draft terminal, bonus terminal, normal commit) must enter LaneGuard::enter_for_slot"
        );
    }
}
