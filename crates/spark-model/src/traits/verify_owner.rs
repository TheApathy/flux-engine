// SPDX-License-Identifier: AGPL-3.0-only

//! Generation-bearing ownership of the model's shared verify scratch.
//!
//! A scheduler frame must retain the returned generation-bearing ticket until
//! commit or abort completes. Failed commits preserve it for an abort/retry.

#[path = "verify_batch_owner.rs"]
pub(crate) mod batch;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Owner {
    pub slot: usize,
    pub session: u64,
    pub cursor: usize,
    pub rows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticket {
    owner: Owner,
    generation: u64,
}

impl Ticket {
    pub(crate) fn owner(self) -> Owner {
        self.owner
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Frame {
    ticket: Ticket,
    accepted: Option<usize>,
    batch: Option<batch::LaneTicket>,
}

impl Frame {
    pub(crate) fn new(ticket: Ticket) -> Self {
        Self {
            ticket,
            accepted: None,
            batch: None,
        }
    }
    pub(crate) fn new_batch(lane: batch::LaneTicket) -> Self {
        Self {
            ticket: lane.ticket(),
            accepted: None,
            batch: Some(lane),
        }
    }
    pub(crate) fn batch(self) -> Option<batch::LaneTicket> {
        self.batch
    }
    pub(crate) fn ticket(self) -> Ticket {
        self.ticket
    }
    pub(crate) fn active(self) -> bool {
        self.accepted.is_none()
    }
    pub(crate) fn accepted(self) -> Option<usize> {
        self.accepted
    }
    pub(crate) fn committed(&mut self, accepted: usize) {
        self.accepted = Some(accepted);
    }

    pub(crate) fn matches_sequence(
        self,
        slot: usize,
        session: u64,
        cursor: usize,
        rows: usize,
    ) -> bool {
        let owner = self.ticket.owner;
        owner.slot == slot
            && owner.session == session
            && owner.rows == rows
            && cursor >= owner.cursor
            && cursor <= owner.cursor + owner.rows
    }
}

/// Metadata for sequence-owned hidden rows. A mismatched consumer cannot
/// erase the rightful publication merely by presenting the same row count.
#[derive(Default)]
pub(crate) struct StagedFrame(Option<(Ticket, usize)>);

impl StagedFrame {
    pub(crate) fn bind(&self, owner: Option<Ticket>) -> Result<(), Failure> {
        match self.0 {
            Some((published, _)) if Some(published) != owner => Err(Failure::StaleOwner),
            _ => Ok(()),
        }
    }

    pub(crate) fn check_publish(&self, owner: Ticket, rows: usize) -> Result<(), Failure> {
        if rows >= owner.owner.rows {
            return Err(Failure::BadGeometry);
        }
        match self.0 {
            Some(published) if published != (owner, rows) => Err(Failure::StaleOwner),
            _ => Ok(()),
        }
    }

    pub(crate) fn publish(&mut self, owner: Ticket, rows: usize) -> Result<(), Failure> {
        self.check_publish(owner, rows)?;
        self.0 = Some((owner, rows));
        Ok(())
    }

    pub(crate) fn take(&mut self, owner: Option<Ticket>, rows: usize) -> Result<bool, Failure> {
        match (owner, self.0) {
            (_, None) => Ok(false),
            (Some(owner), Some(observed)) if observed == (owner, rows) => {
                self.0 = None;
                Ok(true)
            }
            _ => Err(Failure::StaleOwner),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.0 = None;
    }
    pub(crate) fn pending(&self) -> bool {
        self.0.is_some()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    Poisoned,
    AlreadyOwned,
    StaleOwner,
    BadGeometry,
    GenerationExhausted,
}

#[derive(Default)]
pub struct Transactions {
    generation: u64,
    active: Option<Ticket>,
    poisoned: bool,
}

impl Transactions {
    pub(crate) fn idle(&self) -> bool {
        self.active.is_none()
    }

    pub(crate) fn require_idle(&self) -> Result<(), Failure> {
        self.check()?;
        if self.idle() {
            Ok(())
        } else {
            Err(Failure::AlreadyOwned)
        }
    }

    // Cleanup may still drain the rightful frame after a sticky owner fault.
    // It must never restore another ticket's shared journal.
    pub(crate) fn owns_for_cleanup(&self, ticket: Ticket) -> bool {
        self.active == Some(ticket)
    }

    pub(crate) fn validate(&mut self, ticket: Ticket, owner: Owner) -> Result<(), Failure> {
        self.require(ticket)?;
        if ticket.owner == owner {
            Ok(())
        } else {
            self.fail(Failure::StaleOwner)
        }
    }

    pub fn check(&self) -> Result<(), Failure> {
        if self.poisoned {
            Err(Failure::Poisoned)
        } else {
            Ok(())
        }
    }

    fn fail<T>(&mut self, reason: Failure) -> Result<T, Failure> {
        self.poisoned = true;
        Err(reason)
    }

    pub fn begin(&mut self, owner: Owner) -> Result<Ticket, Failure> {
        self.check()?;
        if owner.rows == 0 || owner.cursor.checked_add(owner.rows).is_none() {
            return self.fail(Failure::BadGeometry);
        }
        if self.active.is_some() {
            return self.fail(Failure::AlreadyOwned);
        }
        let Some(generation) = self.generation.checked_add(1) else {
            return self.fail(Failure::GenerationExhausted);
        };
        let ticket = Ticket { owner, generation };
        self.generation = generation;
        self.active = Some(ticket);
        Ok(ticket)
    }

    fn require(&mut self, ticket: Ticket) -> Result<(), Failure> {
        self.check()?;
        if self.active == Some(ticket) {
            Ok(())
        } else {
            self.fail(Failure::StaleOwner)
        }
    }

    pub fn commit_succeeded(
        &mut self,
        ticket: Ticket,
        accepted_rows: usize,
    ) -> Result<(), Failure> {
        self.require(ticket)?;
        if accepted_rows > ticket.owner.rows {
            return self.fail(Failure::BadGeometry);
        }
        self.active = None;
        Ok(())
    }

    pub fn commit_failed(&mut self, ticket: Ticket) -> Result<(), Failure> {
        // Keep the exact owner for the failed commit's subsequent abort/retry.
        self.require(ticket)
    }

    pub fn abort_succeeded(&mut self, ticket: Ticket) -> Result<(), Failure> {
        self.require(ticket)?;
        self.active = None;
        Ok(())
    }

    pub fn abort_failed(&mut self, ticket: Ticket) -> Result<(), Failure> {
        self.require(ticket)?;
        self.fail(Failure::Poisoned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owner(slot: usize, rows: usize) -> Owner {
        Owner {
            slot,
            session: 17 + slot as u64,
            cursor: 2049 + slot,
            rows,
        }
    }

    #[test]
    fn every_short_and_wide_acceptance_releases_only_its_ticket() {
        for rows in 2..=5 {
            for accepted in 0..=rows {
                let mut ledger = Transactions::default();
                let ticket = ledger.begin(owner(0, rows)).unwrap();
                ledger.commit_succeeded(ticket, accepted).unwrap();
                let next = ledger.begin(owner(1, rows)).unwrap();
                assert_ne!(ticket, next);
                ledger.abort_succeeded(next).unwrap();
            }
        }
    }

    #[test]
    fn failed_commit_retains_owner_until_successful_abort() {
        let mut ledger = Transactions::default();
        let ticket = ledger.begin(owner(0, 5)).unwrap();
        ledger.commit_failed(ticket).unwrap();
        assert_eq!(ledger.active, Some(ticket));
        ledger.abort_succeeded(ticket).unwrap();
        assert!(ledger.begin(owner(1, 2)).is_ok());
    }

    #[test]
    fn failed_commit_can_retry_with_same_ticket() {
        let mut ledger = Transactions::default();
        let ticket = ledger.begin(owner(0, 5)).unwrap();
        ledger.commit_failed(ticket).unwrap();
        ledger.commit_succeeded(ticket, 3).unwrap();
        assert!(ledger.check().is_ok());
    }

    #[test]
    fn overlapping_request_cannot_take_shared_journal() {
        let mut ledger = Transactions::default();
        let ticket = ledger.begin(owner(0, 5)).unwrap();
        assert_eq!(ledger.begin(owner(1, 5)), Err(Failure::AlreadyOwned));
        assert_eq!(ledger.abort_succeeded(ticket), Err(Failure::Poisoned));
        assert_eq!(ledger.active, Some(ticket));
    }

    #[test]
    fn slot_reuse_with_same_session_cursor_cannot_reuse_old_generation() {
        let mut ledger = Transactions::default();
        let old = ledger.begin(owner(0, 5)).unwrap();
        ledger.abort_succeeded(old).unwrap();
        let current = ledger.begin(owner(0, 5)).unwrap();
        assert_ne!(old, current);
        assert_eq!(ledger.commit_succeeded(old, 1), Err(Failure::StaleOwner));
        assert_eq!(ledger.active, Some(current));
    }

    #[test]
    fn wrong_slot_session_cursor_or_rows_cannot_abort_another_frame() {
        for field in 0..4 {
            let mut ledger = Transactions::default();
            let ticket = ledger.begin(owner(0, 5)).unwrap();
            let mut forged = ticket;
            match field {
                0 => forged.owner.slot += 1,
                1 => forged.owner.session += 1,
                2 => forged.owner.cursor += 1,
                _ => forged.owner.rows -= 1,
            }
            assert_eq!(ledger.abort_succeeded(forged), Err(Failure::StaleOwner));
            assert_eq!(ledger.active, Some(ticket));
        }
    }

    #[test]
    fn failed_abort_poison_is_sticky_and_preserves_unresolved_owner() {
        let mut ledger = Transactions::default();
        let ticket = ledger.begin(owner(0, 5)).unwrap();
        assert_eq!(ledger.abort_failed(ticket), Err(Failure::Poisoned));
        assert_eq!(ledger.begin(owner(1, 2)), Err(Failure::Poisoned));
        assert_eq!(ledger.commit_succeeded(ticket, 1), Err(Failure::Poisoned));
        assert_eq!(ledger.active, Some(ticket));
    }

    #[test]
    fn excess_acceptance_poisons_without_releasing_journal() {
        let mut ledger = Transactions::default();
        let ticket = ledger.begin(owner(0, 2)).unwrap();
        assert_eq!(
            ledger.commit_succeeded(ticket, 3),
            Err(Failure::BadGeometry)
        );
        assert_eq!(ledger.active, Some(ticket));
    }

    #[test]
    fn invalid_geometry_and_generation_overflow_fail_closed() {
        for rows in [0, usize::MAX] {
            let mut ledger = Transactions::default();
            assert_eq!(ledger.begin(owner(0, rows)), Err(Failure::BadGeometry));
            assert!(ledger.active.is_none());
        }
        let mut ledger = Transactions {
            generation: u64::MAX,
            ..Default::default()
        };
        assert_eq!(ledger.begin(owner(0, 5)), Err(Failure::GenerationExhausted));
        assert!(ledger.active.is_none());
    }

    #[test]
    fn k1_shadow_restore_retains_generation_until_both_restores_finish() {
        let mut ledger = Transactions::default();
        let original = owner(0, 5);
        let ticket = ledger.begin(original).unwrap();
        let frame = Frame::new(ticket);
        // The scheduler rewinds its host cursor before restoring QSA and
        // recurrent state. Each intermediate point still owns this frame.
        for cursor in [original.cursor + 5, original.cursor] {
            assert!(frame.matches_sequence(original.slot, original.session, cursor, 5));
            ledger.validate(ticket, original).unwrap();
            assert_eq!(ledger.active, Some(ticket));
        }
        // Only the finish called after both successful restores releases it.
        ledger.abort_succeeded(ticket).unwrap();
        let production = ledger.begin(original).unwrap();
        assert_ne!(production, ticket);
        assert_eq!(ledger.active, Some(production));
    }

    #[test]
    fn k1_serial_rebuild_keeps_actual_width_and_commits_only_selected_prefix() {
        for rows in [2, 3, 4, 5, 17] {
            for accepted in 1..=rows {
                let mut ledger = Transactions::default();
                let original = owner(0, rows);
                let ticket = ledger.begin(original).unwrap();
                let mut frame = Frame::new(ticket);
                for rebuilt in 0..=accepted {
                    assert!(frame.matches_sequence(
                        0,
                        original.session,
                        original.cursor + rebuilt,
                        rows
                    ));
                    ledger.validate(ticket, original).unwrap();
                    assert_eq!(ledger.active, Some(ticket));
                }
                ledger.commit_succeeded(ticket, accepted).unwrap();
                frame.committed(accepted);
                assert!(!frame.active());
                assert_eq!(frame.accepted(), Some(accepted));
                assert!(ledger.idle());
            }
        }
    }

    #[test]
    fn failed_rebuild_and_failed_commit_do_not_invent_a_new_retry_generation() {
        let mut ledger = Transactions::default();
        let original = owner(0, 5);
        let ticket = ledger.begin(original).unwrap();
        let mut frame = Frame::new(ticket);
        for _ in 0..3 {
            ledger.commit_failed(ticket).unwrap();
            assert_eq!(ledger.active, Some(ticket));
            assert_eq!(ledger.generation, ticket.generation);
            assert!(frame.active());
            assert!(frame.matches_sequence(0, original.session, original.cursor, 5));
        }
        ledger.commit_succeeded(ticket, 3).unwrap();
        frame.committed(3);
        assert_eq!(frame.accepted(), Some(3));
        assert!(ledger.idle());
        assert!(ledger.begin(original).unwrap().generation > ticket.generation);
    }

    #[test]
    fn wrong_geometry_cannot_reinterpret_a_wide_ticket_as_a_short_frame() {
        let mut ledger = Transactions::default();
        let original = owner(0, 5);
        let frame = Frame::new(ledger.begin(original).unwrap());
        assert!(!frame.matches_sequence(0, original.session, original.cursor, 1));
        assert!(!frame.matches_sequence(0, original.session + 1, original.cursor, 5));
        assert!(!frame.matches_sequence(1, original.session, original.cursor, 5));
        assert!(!frame.matches_sequence(0, original.session, original.cursor - 1, 5));
        assert!(!frame.matches_sequence(0, original.session, original.cursor + 6, 5));
    }

    #[test]
    fn staged_rows_reject_equal_count_from_every_foreign_ticket_field() {
        for field in 0..5 {
            let mut ledger = Transactions::default();
            let ticket = ledger.begin(owner(0, 5)).unwrap();
            let mut staged = StagedFrame::default();
            staged.publish(ticket, 3).unwrap();
            let mut foreign = ticket;
            match field {
                0 => foreign.owner.slot += 1,
                1 => foreign.owner.session += 1,
                2 => foreign.owner.cursor += 1,
                3 => foreign.owner.rows += 1,
                _ => foreign.generation += 1,
            }
            assert_eq!(staged.take(Some(foreign), 3), Err(Failure::StaleOwner));
            assert_eq!(staged.bind(Some(foreign)), Err(Failure::StaleOwner));
            assert_eq!(staged.publish(foreign, 3), Err(Failure::StaleOwner));
            assert_eq!(staged.take(Some(ticket), 3), Ok(true));
            assert_eq!(staged.take(Some(ticket), 3), Ok(false));
        }
    }

    #[test]
    fn staged_width_and_missing_ticket_fail_without_erasing_rightful_rows() {
        let mut ledger = Transactions::default();
        let ticket = ledger.begin(owner(0, 5)).unwrap();
        let mut staged = StagedFrame::default();
        assert_eq!(staged.publish(ticket, 5), Err(Failure::BadGeometry));
        staged.publish(ticket, 3).unwrap();
        assert_eq!(staged.take(Some(ticket), 2), Err(Failure::StaleOwner));
        assert_eq!(staged.take(None, 3), Err(Failure::StaleOwner));
        assert_eq!(staged.take(Some(ticket), 3), Ok(true));
    }

    #[test]
    fn deferred_replay_retains_generation_after_commit_handoff_is_retired() {
        let mut ledger = Transactions::default();
        let original = owner(0, 5);
        let ticket = ledger.begin(original).unwrap();
        ledger.commit_succeeded(ticket, 4).unwrap();
        let pending_owner = Some(ticket);
        let mut staged = StagedFrame::default();
        staged.publish(ticket, 3).unwrap();
        let reused = ledger.begin(original).unwrap();
        assert_ne!(reused, ticket);
        assert_eq!(staged.take(Some(reused), 3), Err(Failure::StaleOwner));
        assert_eq!(staged.take(pending_owner, 3), Ok(true));
        staged.publish(reused, 3).unwrap();
        staged.clear();
        assert_eq!(staged.take(Some(reused), 3), Ok(false));
    }

    #[test]
    fn rightful_cleanup_remains_identifiable_after_sticky_owner_fault() {
        let mut ledger = Transactions::default();
        let ticket = ledger.begin(owner(0, 5)).unwrap();
        assert_eq!(ledger.begin(owner(1, 5)), Err(Failure::AlreadyOwned));
        assert!(ledger.owns_for_cleanup(ticket));
        let mut wrong = ticket;
        wrong.owner.slot += 1;
        assert!(!ledger.owns_for_cleanup(wrong));
        // Raw cleanup is still allowed, but the poisoned ledger cannot
        // certify success or admit another frame in this process.
        assert_eq!(ledger.abort_succeeded(ticket), Err(Failure::Poisoned));
        assert_eq!(ledger.active, Some(ticket));
        assert_eq!(ledger.check(), Err(Failure::Poisoned));
    }

    #[test]
    fn stale_completed_ticket_poisons_idle_workload_admission() {
        let mut ledger = Transactions::default();
        let ticket = ledger.begin(owner(0, 5)).unwrap();
        ledger.commit_succeeded(ticket, 3).unwrap();
        assert!(ledger.idle());
        assert_eq!(ledger.require_idle(), Ok(()));
        assert_eq!(ledger.abort_succeeded(ticket), Err(Failure::StaleOwner));
        assert!(ledger.idle());
        assert_eq!(ledger.require_idle(), Err(Failure::Poisoned));
        assert_eq!(ledger.begin(owner(1, 5)), Err(Failure::Poisoned));
    }

    #[test]
    fn committed_handoff_admission_cannot_erase_unconsumed_publication() {
        let mut ledger = Transactions::default();
        let ticket = ledger.begin(owner(0, 5)).unwrap();
        ledger.commit_succeeded(ticket, 3).unwrap();
        let mut frame = Frame::new(ticket);
        frame.committed(3);
        let mut staged = StagedFrame::default();
        staged.publish(ticket, 2).unwrap();
        assert!(ledger.require_idle().is_ok());
        assert!(!frame.active());
        assert!(staged.pending());
        // Model begin rejects while this production predicate is true;
        // neither failed admission nor a foreign consumer clears the note.
        assert_eq!(staged.take(None, 2), Err(Failure::StaleOwner));
        assert!(staged.pending());
        assert_eq!(staged.take(Some(frame.ticket()), 2), Ok(true));
        assert!(!staged.pending());
    }

    #[test]
    fn retire_committed_verify_handoff_clears_committed_and_preserves_active_frames() {
        let mut ledger = Transactions::default();
        let active_ticket = ledger.begin(owner(0, 5)).unwrap();
        let mut seq = super::super::SequenceState::for_model_owned_state(0);
        seq.verify_frame = Some(Frame::new(active_ticket));
        seq.retire_committed_verify_handoff();
        assert!(
            seq.verify_frame.is_some_and(|frame| frame.active()),
            "active unresolved verify frames must never be erased by skipped-propose handoff retirement"
        );

        ledger.commit_succeeded(active_ticket, 3).unwrap();
        seq.verify_frame.as_mut().unwrap().committed(3);
        assert!(seq.verify_frame.is_some_and(|frame| !frame.active()));
        seq.retire_committed_verify_handoff();
        assert!(
            seq.verify_frame.is_none(),
            "committed post-verify handoff must retire cleanly when proposing is skipped"
        );
    }
}
