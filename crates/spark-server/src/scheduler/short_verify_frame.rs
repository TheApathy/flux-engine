// SPDX-License-Identifier: AGPL-3.0-only

//! Transaction boundaries shared by the fixed-width K3/K4 scheduler paths.

use super::*;

pub(super) struct ShortVerifyFrame {
    pre_verify_len: usize,
    pre_verify_tokens: usize,
    consumed_rows: usize,
    k: usize,
    /// Phase and output length at frame entry, for the settled-frame receipt.
    thinking_entry: bool,
    emitted_before: usize,
}

impl ShortVerifyFrame {
    pub(super) fn capture(a: &ActiveSeq, k: usize) -> Self {
        Self {
            pre_verify_len: a.seq.seq_len,
            pre_verify_tokens: a.seq.tokens.len(),
            consumed_rows: 0,
            k,
            thinking_entry: a.inside_thinking,
            emitted_before: a.output_tokens.len(),
        }
    }

    fn receipt(
        &self,
        a: &ActiveSeq,
        accepted: usize,
        terminal: bool,
    ) -> spec_accept_log::CommittedFrameReceipt {
        spec_accept_log::CommittedFrameReceipt {
            verified_rows: self.k,
            drafted: self.k.saturating_sub(1),
            thinking_entry: self.thinking_entry,
            consumed_rows: self.consumed_rows,
            accepted,
            emitted: a.output_tokens.len().saturating_sub(self.emitted_before),
            terminal,
        }
    }

    #[cfg(test)]
    pub(super) fn receipt_for_test(
        &self,
        a: &ActiveSeq,
        accepted: usize,
        terminal: bool,
    ) -> spec_accept_log::CommittedFrameReceipt {
        self.receipt(a, accepted, terminal)
    }

    /// The frame's state commit, hidden save and proposer trim succeeded.
    pub(super) fn settled(&self, a: &ActiveSeq, accepted: usize) {
        spec_accept_log::record_settled_frame(&self.receipt(a, accepted, false));
    }

    pub(super) fn emit(
        &mut self,
        a: &mut ActiveSeq,
        token: u32,
        logprobs: Option<crate::api::TokenLogprobs>,
    ) {
        if emit_token_for_verify(a, token, logprobs) {
            self.consumed_rows += 1;
        }
    }

    pub(super) fn fail(
        &self,
        model: &dyn Model,
        a: &mut ActiveSeq,
        stage: &str,
        error: impl std::fmt::Display,
    ) {
        a.seq.seq_len = self.pre_verify_len;
        a.seq.tokens.truncate(self.pre_verify_tokens);
        // Drain and restore both state owners even after a partially enqueued
        // commit. Notification does not free; retirement remains single-owner.
        let error = match rollback_failed_verify_frame(model, &mut a.seq, self.k) {
            Ok(()) => format!("{error:#}"),
            Err(rollback_error) => {
                tracing::error!("short verify abort rollback failed: {rollback_error:#}");
                format!("{error:#}; verify abort rollback failed: {rollback_error:#}")
            }
        };
        fail_speculative_step(a, stage, error);
    }

    /// Run only before the ordinary commit consumes this frame's QSA journal.
    pub(super) fn reconcile_terminal(
        &self,
        model: &dyn Model,
        a: &mut ActiveSeq,
        accepted: usize,
        tokens: &[u32],
    ) {
        let mut abort_attempted = false;
        let result = (|| -> Result<()> {
            anyhow::ensure!(tokens.len() == self.k, "short verify frame width changed");
            // Count classified rows, not client-visible tokens: silent think
            // tags and suppressed EOS still belong to the target input prefix.
            let plan = terminal_verify_commit_plan(
                self.pre_verify_len,
                a.seq.seq_len,
                a.seq.tokens.len(),
                self.k,
                accepted,
                0,
                self.consumed_rows,
                None,
            )?;
            a.seq.tokens.truncate(plan.target_seq_len);
            a.seq.seq_len = plan.target_seq_len;
            if plan.committed_inputs == 0 {
                abort_attempted = true;
                rollback_failed_verify_frame(model, &mut a.seq, self.k)?;
            } else {
                model.commit_verify_state_async(&mut a.seq, plan.committed_inputs, self.k)?;
            }
            Ok(())
        })();
        if result.is_ok() {
            spec_accept_log::record_settled_frame(&self.receipt(a, accepted, true));
        }
        if let Err(error) = result {
            if abort_attempted {
                // The zero-row plan already restored the host extent and
                // attempted both backend owners. Do not retry an armed journal.
                fail_speculative_step(a, "short terminal verify abort", error);
            } else {
                self.fail(model, a, "short terminal verify commit", error);
            }
        }
    }
}

#[cfg(test)]
#[path = "short_verify_frame_tests.rs"]
mod tests;
