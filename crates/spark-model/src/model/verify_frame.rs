// SPDX-License-Identifier: AGPL-3.0-only

use super::types::TransformerModel;
use crate::traits::SequenceState;
use crate::traits::verify_owner::{Failure, Frame, Owner, Ticket};
use anyhow::{Result, ensure};

pub(super) fn protocol_error(context: &'static str, error: Failure) -> anyhow::Error {
    // The ledger lock may still be held. Never re-lock it from this bridge.
    crate::model::verify_failure::poison_owner();
    anyhow::anyhow!("{context}: {error:?}")
}

pub(super) fn admission_error(context: &'static str, error: Failure) -> anyhow::Error {
    // Busy admission is routine; a poisoned ledger is a process-fatal fault.
    if error == Failure::Poisoned {
        crate::model::verify_failure::poison_owner();
    }
    anyhow::anyhow!("{context}: {error:?}")
}

impl TransformerModel {
    pub(super) fn begin_verify_frame(&self, seq: &mut SequenceState, rows: usize) -> Result<()> {
        crate::model::verify_failure::check()?;
        if !self.config.is_qwen4_exp() {
            return Ok(());
        }
        if rows <= 1 {
            self.ensure_verify_sequence_idle(seq)?;
            self.qwen4_seqstate_cache_save_if_due(seq);
            return self.qwen4_seqstate_roundtrip_if_due(seq);
        }
        if seq.verify_frame.is_some_and(|frame| !frame.active()) {
            for state in [&seq.proposer_state, &seq.proposer_state_alt] {
                let pending = state
                    .as_ref()
                    .and_then(|state| {
                        state
                            .as_any()
                            .downcast_ref::<crate::layers::qwen4_mtp::Qwen4MtpState>()
                    })
                    .is_some_and(|native| native.has_unconsumed_verify_frame());
                ensure!(
                    !pending,
                    "committed native MTP handoff has not been consumed"
                );
            }
        }
        self.qwen4_seqstate_cache_save_if_due(seq);
        self.qwen4_seqstate_roundtrip_if_due(seq)?;
        let owner = Owner {
            slot: seq.slot_idx,
            session: seq.session_hash,
            cursor: seq.seq_len,
            rows,
        };
        let ticket = self
            .verify_transactions
            .lock()
            .begin(owner)
            .map_err(|error| protocol_error("verify frame begin failed", error))?;
        seq.verify_frame = Some(Frame::new(ticket));
        Ok(())
    }

    fn frame_ticket(
        &self,
        seq: &SequenceState,
        rows: usize,
        cleanup: bool,
    ) -> Result<Option<Ticket>> {
        if !self.config.is_qwen4_exp() {
            return Ok(None);
        }
        if rows <= 1 {
            if cleanup {
                ensure!(
                    !seq.verify_frame.is_some_and(|frame| frame.active())
                        && self.verify_transactions.lock().idle(),
                    "short cleanup cannot restore an unresolved wide verify frame"
                );
            } else {
                self.ensure_verify_sequence_idle(seq)?;
            }
            return Ok(None);
        }
        let frame = seq.verify_frame.filter(|frame| frame.active());
        if cleanup && frame.is_none() {
            ensure!(
                self.verify_transactions.lock().idle(),
                "another verify frame owns shared cleanup"
            );
            return Ok(None);
        }
        let ticket = frame
            .ok_or_else(|| anyhow::anyhow!("verify frame has no active sequence ticket"))?
            .ticket();
        ensure!(
            frame.unwrap().batch().is_none(),
            "batch lane cannot use the scalar root ticket"
        );
        let owner = ticket.owner();
        ensure!(
            frame
                .unwrap()
                .matches_sequence(seq.slot_idx, seq.session_hash, seq.seq_len, rows),
            "verify frame slot/session/cursor/width mismatch"
        );
        let mut ledger = self.verify_transactions.lock();
        if cleanup {
            ensure!(
                ledger.owns_for_cleanup(ticket),
                "cleanup ticket does not own the shared verify frame"
            );
        } else {
            ledger
                .validate(ticket, owner)
                .map_err(|error| protocol_error("verify frame owner failed", error))?;
        }
        Ok(Some(ticket))
    }

    pub(super) fn ensure_verify_sequence_idle(&self, seq: &SequenceState) -> Result<()> {
        crate::model::verify_failure::check()?;
        if self.config.is_qwen4_exp() {
            ensure!(
                !seq.verify_frame.is_some_and(|frame| frame.active()),
                "sequence still owns an unresolved verify frame"
            );
            self.verify_transactions
                .lock()
                .require_idle()
                .map_err(|error| admission_error("verify workload idle admission failed", error))?;
        }
        Ok(())
    }

    pub(super) fn ensure_verify_healthy(&self) -> Result<()> {
        crate::model::verify_failure::check()?;
        if self.config.is_qwen4_exp() {
            self.verify_transactions
                .lock()
                .check()
                .map_err(|error| admission_error("verify workload admission failed", error))?;
        }
        Ok(())
    }

    pub(super) fn discard_native_verify_frame(&self, seq: &mut SequenceState) {
        for state in [&mut seq.proposer_state, &mut seq.proposer_state_alt] {
            if let Some(native) = state.as_mut().and_then(|state| {
                state
                    .as_any_mut()
                    .downcast_mut::<crate::layers::qwen4_mtp::Qwen4MtpState>()
            }) {
                native.discard_verify_frame();
            }
        }
    }

    pub(super) fn commit_ep_short_owned(
        &self,
        seq: &mut SequenceState,
        accepted_drafts: usize,
        rows: usize,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        let accepted = accepted_drafts
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("EP accepted width overflow"))?;
        ensure!(
            accepted <= rows && seq.seq_len >= rows - accepted,
            "EP short verify acceptance is outside its actual width"
        );
        self.frame_ticket(seq, rows, false)?;
        seq.seq_len -= rows - accepted;
        seq.tokens.truncate(seq.seq_len);
        self.commit_owned_verify_state(seq, accepted, rows, accepted - 1)?;
        self.trim_proposer_state_dispatch(seq, accepted_drafts, 0)
    }

    pub(super) fn commit_owned_verify_state(
        &self,
        seq: &mut SequenceState,
        accepted: usize,
        rows: usize,
        last_slot: usize,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        if let Some(lane) = self.batch_lane(seq, rows, false)? {
            return self.commit_owned_batch_state(seq, lane, accepted, rows, last_slot);
        }
        let ticket = self.frame_ticket(seq, rows, false)?;
        ensure!(
            accepted <= rows,
            "verify acceptance exceeds the actual physical width"
        );
        let result = self.commit_verify_state_async_dispatch(seq, accepted, rows, last_slot);
        if let Some(ticket) = ticket {
            let mut ledger = self.verify_transactions.lock();
            if result.is_ok() {
                ledger
                    .commit_succeeded(ticket, accepted)
                    .map_err(|error| protocol_error("verify commit owner failed", error))?;
                seq.verify_frame.as_mut().unwrap().committed(accepted);
            } else {
                // Sampled DFlash may retry after irreversible output. Preserve
                // this exact generation and every diagnostic route source.
                ledger
                    .commit_failed(ticket)
                    .map_err(|error| protocol_error("verify failed-commit owner failed", error))?;
            }
        }
        result
    }

    pub(super) fn restore_owned_qsa_prefix(
        &self,
        seq: &SequenceState,
        accepted: usize,
        rows: usize,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.frame_ticket(seq, rows, false)?;
        ensure!(
            accepted <= rows,
            "QSA acceptance exceeds the actual physical width"
        );
        // Diagnostics restore only QSA, then rebuild K1 state or replay the
        // same production frame. They retain the active generation throughout.
        self.commit_qwen4_qsa_k5_only_dispatch(accepted, rows)
    }

    pub(super) fn finish_verify_diagnostic(
        &self,
        seq: &mut SequenceState,
        rows: usize,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        if let Some(ticket) = self.frame_ticket(seq, rows, false)? {
            self.verify_transactions
                .lock()
                .abort_succeeded(ticket)
                .map_err(|error| protocol_error("verify diagnostic owner failed", error))?;
            seq.verify_frame = None;
        }
        Ok(())
    }

    pub(super) fn finish_verify_rebuild(
        &self,
        seq: &mut SequenceState,
        accepted: usize,
        rows: usize,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        if let Some(ticket) = self.frame_ticket(seq, rows, false)? {
            self.verify_transactions
                .lock()
                .commit_succeeded(ticket, accepted)
                .map_err(|error| protocol_error("verify rebuilt owner failed", error))?;
            seq.verify_frame.as_mut().unwrap().committed(accepted);
        }
        Ok(())
    }

    pub(super) fn abort_owned_verify_state(
        &self,
        seq: &mut SequenceState,
        rows: usize,
    ) -> Result<()> {
        if let Some(lane) = self.batch_lane(seq, rows, true)? {
            return self.abort_owned_batch_lane(seq, lane);
        }
        let ticket = self.frame_ticket(seq, rows, true)?;
        // The raw abort still attempts secondary drain, QSA and recurrent/PLE
        // cleanup independently. Its fatal-poison policy is owned separately.
        let result = self.abort_verify_state_dispatch(seq, rows);
        self.discard_native_verify_frame(seq);
        if let Some(ticket) = ticket {
            let mut ledger = self.verify_transactions.lock();
            if result.is_ok() {
                ledger
                    .abort_succeeded(ticket)
                    .map_err(|error| protocol_error("verify abort owner failed", error))?;
                seq.verify_frame = None;
            } else if let Err(error) = ledger.abort_failed(ticket) {
                let _ = protocol_error("verify abort failure owner failed", error);
            }
        }
        result
    }

    pub(super) fn free_owned_sequence(&self, seq: &mut SequenceState) -> Result<()> {
        let abort = if let Some(frame) = seq.verify_frame.filter(|frame| frame.active()) {
            self.abort_owned_verify_state(seq, frame.ticket().owner().rows)
        } else {
            Ok(())
        };
        // Free remains callable after fatal cleanup errors and is attempted
        // even when the rightful frame could not be restored.
        let free = self.free_sequence_dispatch(seq);
        self.discard_native_verify_frame(seq);
        if abort.is_ok() && free.is_ok() {
            seq.verify_frame = None;
        }
        abort.and(free)
    }
}
