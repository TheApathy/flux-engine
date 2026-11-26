// SPDX-License-Identifier: AGPL-3.0-only

//! One shared scratch reservation across a bounded cohort of K-row lane
//! owners. In default fixed-K5 mode every owner verifies `K=5`; under
//! `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1` each owner independently carries a
//! ragged width `K_s in 1..=8` (`K_s == 1` is anchor-only with zero drafts)
//! packed into 256-row frames (`MAX_FRAME_ROWS`) of up to 64 sequences.
//! Device operations and the existing process fault latch remain the model's
//! responsibility. This core never certifies that an enqueue or cleanup ran.

use super::{Failure, Owner, Ticket, Transactions};

/// Physical verify lanes (one per SSM slot) and cohort owners. The fixed-K5
/// server admits C2-C32; the adaptive-K server admits C2-C64 per frame, up to
/// 128 physical lanes across multi-frame cohorts.
pub(crate) const MAX_LANES: usize = 128;
pub(crate) const MAX_FRAME_SEQS: usize = 64;
pub(crate) const MAX_FRAME_ROWS: usize = 256;
/// The fixed verify width of the default (non-adaptive) cohort.
pub(crate) const K: usize = 5;
/// Narrowest and widest per-lane cohort widths (`K=1` anchor-only through
/// `K=8`, where 32 x K8 fits inside one 256-row frame).
pub(crate) const MIN_K: usize = 1;
pub(crate) const MAX_K: usize = 8;
pub(crate) const MAX_RANGES_PER_LANE: usize = 512;

/// Whether a per-sequence width `k` is admitted: K5 only by default, K1..K8
/// with `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1`.
pub(crate) const fn width_admitted(k: usize, adaptive: bool) -> bool {
    if adaptive {
        MIN_K <= k && k <= MAX_K
    } else {
        k == K
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Reservation(Ticket);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaneTicket {
    reservation: Reservation,
    ticket: Ticket,
}

impl LaneTicket {
    pub(crate) fn ticket(self) -> Ticket {
        self.ticket
    }
    pub(crate) fn owner(self) -> Owner {
        self.ticket.owner
    }
    pub(crate) fn reservation(self) -> Reservation {
        self.reservation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    pub address: u64,
    pub bytes: usize,
}

impl Region {
    fn end(self) -> Option<u64> {
        if self.address == 0 || self.bytes == 0 {
            return None;
        }
        self.address.checked_add(u64::try_from(self.bytes).ok()?)
    }

    fn overlaps(self, other: Self) -> bool {
        // Callers validate every endpoint before comparing ranges.
        self.address < other.end().unwrap() && other.address < self.end().unwrap()
    }
}

/// A mutable lane span inside an actual model allocation. Shared execution
/// scratch and read-only weights are excluded: the cohort reservation owns
/// scratch, and each subframe supplies its own checked row map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BoundRange {
    pub allocation: Region,
    pub span: Region,
}

impl BoundRange {
    fn valid(self) -> bool {
        match (self.allocation.end(), self.span.end()) {
            (Some(allocation_end), Some(span_end)) => {
                self.span.address >= self.allocation.address && span_end <= allocation_end
            }
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LaneState {
    Reserved,
    Bound,
    InFlight,
    CommittedAwaitingHidden { accepted: usize },
    Published { accepted: usize },
    Retired { accepted: usize },
    Aborted,
}

impl LaneState {
    fn settled(self) -> bool {
        matches!(
            self,
            Self::Published { .. } | Self::Retired { .. } | Self::Aborted
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrameRow {
    pub lane: LaneTicket,
    pub row_base: usize,
}

#[derive(Debug)]
struct Lane {
    ticket: LaneTicket,
    ranges: Vec<BoundRange>,
    state: LaneState,
}

#[derive(Debug)]
struct Cohort {
    reservation: Reservation,
    /// Narrowest and widest per-lane verify widths across this cohort (`1..=8`).
    min_k: usize,
    max_k: usize,
    lanes: Vec<Lane>,
    frame: Option<Vec<LaneTicket>>,
    bound: bool,
}

#[derive(Default, Debug)]
pub(crate) struct BatchTransactions {
    active: Option<Cohort>,
    poisoned: bool,
}

impl BatchTransactions {
    fn poison<T>(&mut self, shared: &mut Transactions, failure: Failure) -> Result<T, Failure> {
        self.poisoned = true;
        // Retain the exact scalar reservation for rightful cleanup. The model
        // must also propagate this failure to its existing process fault latch.
        shared.fail(failure)
    }

    fn healthy(&self, shared: &Transactions) -> Result<(), Failure> {
        if self.poisoned {
            return Err(Failure::Poisoned);
        }
        shared.check()
    }

    fn reservation(&self, shared: &Transactions, reservation: Reservation) -> bool {
        self.active
            .as_ref()
            .is_some_and(|cohort| cohort.reservation == reservation)
            && shared.owns_for_cleanup(reservation.0)
    }

    fn require(
        &mut self,
        shared: &mut Transactions,
        reservation: Reservation,
    ) -> Result<(), Failure> {
        self.healthy(shared)?;
        if self.reservation(shared, reservation) {
            Ok(())
        } else {
            self.poison(shared, Failure::StaleOwner)
        }
    }

    fn index(&self, lane: LaneTicket) -> Option<usize> {
        self.active
            .as_ref()?
            .lanes
            .iter()
            .position(|entry| entry.ticket == lane)
    }

    fn require_lane(
        &mut self,
        shared: &mut Transactions,
        lane: LaneTicket,
    ) -> Result<usize, Failure> {
        self.require(shared, lane.reservation)?;
        match self.index(lane) {
            Some(index) => Ok(index),
            None => self.poison(shared, Failure::StaleOwner),
        }
    }

    /// Reserve once for the whole cohort. Each owner independently carries a
    /// verify width `owner.rows in MIN_K..=MAX_K` (`1..=8`). Shape refusals
    /// and routine busy admission leave the current reservation and scalar
    /// owner intact.
    pub(crate) fn reserve(
        &mut self,
        shared: &mut Transactions,
        owners: &[Owner],
        configured_lanes: usize,
    ) -> Result<(Reservation, Vec<LaneTicket>), Failure> {
        self.healthy(shared)?;
        if self.active.is_some() {
            return Err(Failure::AlreadyOwned);
        }
        shared.require_idle()?;
        if !(2..=MAX_LANES).contains(&owners.len()) || !(2..=MAX_LANES).contains(&configured_lanes)
        {
            return Err(Failure::BadGeometry);
        }
        let mut total_rows = 0usize;
        let mut min_k = usize::MAX;
        let mut max_k = 0usize;
        for (index, owner) in owners.iter().enumerate() {
            if !(MIN_K..=MAX_K).contains(&owner.rows)
                || owner.slot >= configured_lanes
                || owner.cursor.checked_add(owner.rows).is_none()
                || owners[..index].iter().any(|other| other.slot == owner.slot)
            {
                return Err(Failure::BadGeometry);
            }
            let Some(next_rows) = total_rows.checked_add(owner.rows) else {
                return Err(Failure::BadGeometry);
            };
            total_rows = next_rows;
            min_k = min_k.min(owner.rows);
            max_k = max_k.max(owner.rows);
        }
        let root = shared.begin(Owner {
            slot: usize::MAX,
            session: 0,
            cursor: 0,
            rows: total_rows,
        })?;
        let reservation = Reservation(root);
        let tickets: Vec<_> = owners
            .iter()
            .map(|owner| LaneTicket {
                reservation,
                ticket: Ticket {
                    owner: *owner,
                    generation: root.generation,
                },
            })
            .collect();
        self.active = Some(Cohort {
            reservation,
            min_k,
            max_k,
            frame: None,
            bound: false,
            lanes: tickets
                .iter()
                .map(|ticket| Lane {
                    ticket: *ticket,
                    ranges: Vec::new(),
                    state: LaneState::Reserved,
                })
                .collect(),
        });
        Ok((reservation, tickets))
    }

    /// Bind the complete persistent mutable lane set atomically. Every span
    /// must fit its real allocation; all spans must be mutually disjoint,
    /// including spans presented twice within the same lane. A rejected bind
    /// preserves every previous binding and the rightful reservation.
    pub(crate) fn bind(
        &mut self,
        shared: &mut Transactions,
        reservation: Reservation,
        bindings: &[(LaneTicket, Vec<BoundRange>)],
    ) -> Result<(), Failure> {
        self.require(shared, reservation)?;
        let cohort = self.active.as_ref().unwrap();
        if cohort.bound {
            return Err(Failure::AlreadyOwned);
        }
        if bindings.len() != cohort.lanes.len()
            || bindings.iter().enumerate().any(|(i, (ticket, ranges))| {
                *ticket != cohort.lanes[i].ticket
                    || ranges.is_empty()
                    || ranges.len() > MAX_RANGES_PER_LANE
                    || ranges.iter().any(|range| !range.valid())
            })
        {
            return Err(Failure::BadGeometry);
        }
        let mut spans: Vec<_> = bindings
            .iter()
            .flat_map(|(_, ranges)| ranges.iter().map(|range| range.span))
            .collect();
        // Every range passed `BoundRange::valid()`, so every span is a non-empty
        // half-open interval `[address, address + bytes)`. Sorting by start
        // address reduces pairwise disjointness from O(N^2) (~6M checks at C=16)
        // to an exact O(N log N) adjacent sweep.
        spans.sort_unstable_by_key(|span| (span.address, span.bytes));
        if spans.windows(2).any(|pair| pair[0].overlaps(pair[1])) {
            return Err(Failure::BadGeometry);
        }
        let cohort = self.active.as_mut().unwrap();
        for (lane, (_, ranges)) in cohort.lanes.iter_mut().zip(bindings) {
            lane.ranges = ranges.clone();
            lane.state = LaneState::Bound;
        }
        cohort.bound = true;
        Ok(())
    }

    /// Admission for one actual target forward. Validate all row/ticket facts
    /// before marking any lane in flight. C8 K5 is two C4 frames; the cohort's
    /// shared reservation remains held across both frames.
    pub(crate) fn begin_frame(
        &mut self,
        shared: &mut Transactions,
        reservation: Reservation,
        rows: &[FrameRow],
    ) -> Result<(), Failure> {
        self.require(shared, reservation)?;
        let cohort = self.active.as_ref().unwrap();
        if cohort.frame.is_some() {
            return Err(Failure::AlreadyOwned);
        }
        if !cohort.bound || !(2..=MAX_FRAME_SEQS).contains(&rows.len()) {
            return Err(Failure::BadGeometry);
        }
        let mut expected_base = 0usize;
        for (i, row) in rows.iter().enumerate() {
            if row.lane.reservation != reservation
                || row.row_base != expected_base
                || rows[..i].iter().any(|other| other.lane == row.lane)
                || !cohort
                    .lanes
                    .iter()
                    .any(|lane| lane.ticket == row.lane && lane.state == LaneState::Bound)
            {
                return Err(Failure::BadGeometry);
            }
            let Some(next_base) = expected_base.checked_add(row.lane.owner().rows) else {
                return Err(Failure::BadGeometry);
            };
            if next_base > MAX_FRAME_ROWS {
                return Err(Failure::BadGeometry);
            }
            expected_base = next_base;
        }
        let indices: Vec<_> = rows
            .iter()
            .map(|row| self.index(row.lane).unwrap())
            .collect();
        let cohort = self.active.as_mut().unwrap();
        for index in indices {
            cohort.lanes[index].state = LaneState::InFlight;
        }
        cohort.frame = Some(rows.iter().map(|row| row.lane).collect());
        Ok(())
    }

    pub(crate) fn commit_succeeded(
        &mut self,
        shared: &mut Transactions,
        ticket: LaneTicket,
        accepted: usize,
    ) -> Result<(), Failure> {
        let index = self.require_lane(shared, ticket)?;
        if !(1..=ticket.owner().rows).contains(&accepted) {
            return self.poison(shared, Failure::BadGeometry);
        }
        if self.active.as_ref().unwrap().lanes[index].state != LaneState::InFlight {
            return self.poison(shared, Failure::StaleOwner);
        }
        self.active.as_mut().unwrap().lanes[index].state =
            LaneState::CommittedAwaitingHidden { accepted };
        Ok(())
    }

    /// A returned device commit error can hide partial enqueues. Retain the
    /// ticket for cleanup and poison instead of permitting an uncertain retry.
    pub(crate) fn commit_failed(
        &mut self,
        shared: &mut Transactions,
        ticket: LaneTicket,
    ) -> Result<(), Failure> {
        let index = self.require_lane(shared, ticket)?;
        if self.active.as_ref().unwrap().lanes[index].state == LaneState::InFlight {
            self.poison(shared, Failure::Poisoned)
        } else {
            self.poison(shared, Failure::StaleOwner)
        }
    }

    pub(crate) fn publish_hidden(
        &mut self,
        shared: &mut Transactions,
        ticket: LaneTicket,
        accepted: usize,
    ) -> Result<(), Failure> {
        let index = self.require_lane(shared, ticket)?;
        if self.active.as_ref().unwrap().lanes[index].state
            != (LaneState::CommittedAwaitingHidden { accepted })
        {
            return self.poison(shared, Failure::StaleOwner);
        }
        self.active.as_mut().unwrap().lanes[index].state = LaneState::Published { accepted };
        Ok(())
    }

    /// A terminal request cannot propose again, so it has no hidden consumer.
    /// Retire its committed prefix explicitly instead of inventing a publish.
    pub(crate) fn retire_committed(
        &mut self,
        shared: &mut Transactions,
        ticket: LaneTicket,
    ) -> Result<(), Failure> {
        let index = self.require_lane(shared, ticket)?;
        let LaneState::CommittedAwaitingHidden { accepted } =
            self.active.as_ref().unwrap().lanes[index].state
        else {
            return self.poison(shared, Failure::StaleOwner);
        };
        self.active.as_mut().unwrap().lanes[index].state = LaneState::Retired { accepted };
        Ok(())
    }

    pub(crate) fn fail_execution<T>(
        &mut self,
        shared: &mut Transactions,
        reservation: Reservation,
    ) -> Result<T, Failure> {
        self.require(shared, reservation)?;
        self.poison(shared, Failure::Poisoned)
    }

    /// Addressed cleanup remains legal after sticky poison, but no foreign
    /// reservation or subticket can select another lane's mutable storage.
    pub(crate) fn owns_for_cleanup(&self, shared: &Transactions, ticket: LaneTicket) -> bool {
        self.reservation(shared, ticket.reservation) && self.index(ticket).is_some()
    }

    pub(crate) fn state(&self, ticket: LaneTicket) -> Option<LaneState> {
        self.index(ticket)
            .map(|index| self.active.as_ref().unwrap().lanes[index].state)
    }

    #[allow(dead_code)]
    pub(crate) fn ranges(&self, ticket: LaneTicket) -> Option<&[BoundRange]> {
        self.index(ticket)
            .map(|index| self.active.as_ref().unwrap().lanes[index].ranges.as_slice())
    }

    #[allow(dead_code)]
    pub(crate) fn cohort_width_bounds(&self) -> Option<(usize, usize)> {
        self.active
            .as_ref()
            .map(|cohort| (cohort.min_k, cohort.max_k))
    }

    pub(crate) fn frame_row_base(&self, ticket: LaneTicket) -> Option<usize> {
        let cohort = self.active.as_ref()?;
        let frame = cohort.frame.as_ref()?;
        let mut base = 0usize;
        for lane in frame {
            if *lane == ticket {
                return Some(base);
            }
            base = base.checked_add(lane.owner().rows)?;
        }
        None
    }

    pub(crate) fn abort_succeeded(
        &mut self,
        shared: &mut Transactions,
        ticket: LaneTicket,
    ) -> Result<(), Failure> {
        if !self.owns_for_cleanup(shared, ticket) {
            return self.poison(shared, Failure::StaleOwner);
        }
        let index = self.index(ticket).unwrap();
        self.active.as_mut().unwrap().lanes[index].state = LaneState::Aborted;
        Ok(())
    }

    pub(crate) fn abort_failed(
        &mut self,
        shared: &mut Transactions,
        ticket: LaneTicket,
    ) -> Result<(), Failure> {
        if !self.owns_for_cleanup(shared, ticket) {
            return self.poison(shared, Failure::StaleOwner);
        }
        self.poison(shared, Failure::Poisoned)
    }

    /// Attempt every rightful lane even after an earlier callback fails. The
    /// callback performs raw cleanup; callers must not represent a queued or
    /// skipped operation as successful cleanup.
    pub(crate) fn cleanup_all(
        &mut self,
        shared: &mut Transactions,
        reservation: Reservation,
        mut cleanup: impl FnMut(LaneTicket, LaneState) -> Result<(), ()>,
    ) -> Result<(), Failure> {
        if !self.reservation(shared, reservation) {
            return self.poison(shared, Failure::StaleOwner);
        }
        let plan: Vec<_> = self
            .active
            .as_ref()
            .unwrap()
            .lanes
            .iter()
            .map(|lane| (lane.ticket, lane.state))
            .collect();
        let mut failed = false;
        for (ticket, state) in plan {
            if cleanup(ticket, state).is_ok() {
                let _ = self.abort_succeeded(shared, ticket);
            } else {
                failed = true;
                let _ = self.abort_failed(shared, ticket);
            }
        }
        if failed || self.poisoned || shared.check().is_err() {
            Err(Failure::Poisoned)
        } else {
            Ok(())
        }
    }

    pub(crate) fn end_frame(
        &mut self,
        shared: &mut Transactions,
        reservation: Reservation,
    ) -> Result<(), Failure> {
        self.require(shared, reservation)?;
        let cohort = self.active.as_ref().unwrap();
        let frame = cohort.frame.as_ref().ok_or(Failure::AlreadyOwned)?;
        if !frame
            .iter()
            .all(|ticket| self.state(*ticket).is_some_and(LaneState::settled))
        {
            return Err(Failure::AlreadyOwned);
        }
        self.active.as_mut().unwrap().frame = None;
        Ok(())
    }

    pub(crate) fn finish(
        &mut self,
        shared: &mut Transactions,
        reservation: Reservation,
    ) -> Result<(), Failure> {
        self.require(shared, reservation)?;
        let cohort = self.active.as_ref().unwrap();
        if cohort.frame.is_some() || !cohort.lanes.iter().all(|lane| lane.state.settled()) {
            return Err(Failure::AlreadyOwned);
        }
        shared.commit_succeeded(reservation.0, 0)?;
        self.active = None;
        Ok(())
    }

    pub(crate) fn finish_cancelled(
        &mut self,
        shared: &mut Transactions,
        reservation: Reservation,
    ) -> Result<(), Failure> {
        self.require(shared, reservation)?;
        if !self
            .active
            .as_ref()
            .unwrap()
            .lanes
            .iter()
            .all(|lane| lane.state == LaneState::Aborted)
        {
            return Err(Failure::AlreadyOwned);
        }
        self.active.as_mut().unwrap().frame = None;
        self.finish(shared, reservation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owner(slot: usize) -> Owner {
        Owner {
            slot,
            session: 700 + slot as u64,
            cursor: 2049 + slot,
            rows: K,
        }
    }
    fn owners(count: usize) -> Vec<Owner> {
        (0..count).map(owner).collect()
    }
    fn bindings(tickets: &[LaneTicket]) -> Vec<(LaneTicket, Vec<BoundRange>)> {
        tickets
            .iter()
            .map(|ticket| {
                (
                    *ticket,
                    vec![BoundRange {
                        allocation: Region {
                            address: 0x1000,
                            bytes: MAX_LANES * 0x100,
                        },
                        span: Region {
                            address: 0x1000 + ticket.owner().slot as u64 * 0x100,
                            bytes: 0x100,
                        },
                    }],
                )
            })
            .collect()
    }
    fn rows(tickets: &[LaneTicket]) -> Vec<FrameRow> {
        tickets
            .iter()
            .enumerate()
            .map(|(i, lane)| FrameRow {
                lane: *lane,
                row_base: i * K,
            })
            .collect()
    }
    fn reserve(
        count: usize,
    ) -> (
        Transactions,
        BatchTransactions,
        Reservation,
        Vec<LaneTicket>,
    ) {
        let mut shared = Transactions::default();
        let mut batch = BatchTransactions::default();
        let (reservation, tickets) = batch
            .reserve(&mut shared, &owners(count), MAX_LANES)
            .unwrap();
        (shared, batch, reservation, tickets)
    }
    fn bound(
        count: usize,
    ) -> (
        Transactions,
        BatchTransactions,
        Reservation,
        Vec<LaneTicket>,
    ) {
        let (mut shared, mut batch, reservation, tickets) = reserve(count);
        batch
            .bind(&mut shared, reservation, &bindings(&tickets))
            .unwrap();
        (shared, batch, reservation, tickets)
    }
    fn publish(batch: &mut BatchTransactions, shared: &mut Transactions, tickets: &[LaneTicket]) {
        for ticket in tickets {
            batch.commit_succeeded(shared, *ticket, 3).unwrap();
            batch.publish_hidden(shared, *ticket, 3).unwrap();
        }
    }

    #[test]
    fn c8_retains_one_reservation_across_both_c4_subframes_and_publications() {
        let (mut shared, mut batch, reservation, tickets) = bound(8);
        for frame in tickets.chunks(4) {
            batch
                .begin_frame(&mut shared, reservation, &rows(frame))
                .unwrap();
            for ticket in frame {
                batch.commit_succeeded(&mut shared, *ticket, 3).unwrap();
                assert_eq!(
                    batch.end_frame(&mut shared, reservation),
                    Err(Failure::AlreadyOwned)
                );
                batch.publish_hidden(&mut shared, *ticket, 3).unwrap();
            }
            assert_eq!(shared.require_idle(), Err(Failure::AlreadyOwned));
            assert_eq!(
                batch.finish(&mut shared, reservation),
                Err(Failure::AlreadyOwned)
            );
            batch.end_frame(&mut shared, reservation).unwrap();
            assert_eq!(shared.require_idle(), Err(Failure::AlreadyOwned));
        }
        batch.finish(&mut shared, reservation).unwrap();
        assert_eq!(shared.require_idle(), Ok(()));
    }

    #[test]
    fn unbound_and_partial_or_foreign_row_maps_never_mutate_a_lane() {
        let (mut shared, mut batch, reservation, tickets) = reserve(4);
        assert_eq!(
            batch.begin_frame(&mut shared, reservation, &rows(&tickets)),
            Err(Failure::BadGeometry)
        );
        assert!(
            tickets
                .iter()
                .all(|ticket| batch.state(*ticket) == Some(LaneState::Reserved))
        );
        batch
            .bind(&mut shared, reservation, &bindings(&tickets))
            .unwrap();
        for change in 0..3 {
            let mut bad = rows(&tickets);
            if change == 0 {
                bad[1].row_base += 1;
            }
            if change == 1 {
                bad[1].lane = bad[0].lane;
            }
            if change == 2 {
                bad[1].lane.ticket.owner.session += 1;
            }
            assert_eq!(
                batch.begin_frame(&mut shared, reservation, &bad),
                Err(Failure::BadGeometry)
            );
            assert!(
                tickets
                    .iter()
                    .all(|ticket| batch.state(*ticket) == Some(LaneState::Bound))
            );
        }
    }

    #[test]
    fn range_overflow_overlap_and_outside_allocation_preserve_reservation() {
        for change in 0..5 {
            let (mut shared, mut batch, reservation, tickets) = reserve(2);
            let mut bad = bindings(&tickets);
            match change {
                0 => bad[1].1[0].span.address = bad[0].1[0].span.address,
                1 => bad[1].1[0].span.address = u64::MAX - 1,
                2 => bad[1].1[0].span.bytes = 0,
                3 => bad[1].1[0].span.address = 1,
                _ => bad[1].1[0].allocation.address = u64::MAX - 1,
            }
            assert_eq!(
                batch.bind(&mut shared, reservation, &bad),
                Err(Failure::BadGeometry)
            );
            assert!(shared.owns_for_cleanup(reservation.0));
            assert!(
                tickets
                    .iter()
                    .all(|ticket| batch.state(*ticket) == Some(LaneState::Reserved))
            );
            batch
                .bind(&mut shared, reservation, &bindings(&tickets))
                .unwrap();
        }
    }

    #[test]
    fn binding_is_immutable_and_requires_every_exact_subticket() {
        let (mut shared, mut batch, reservation, tickets) = reserve(2);
        let mut missing = bindings(&tickets);
        missing.pop();
        assert_eq!(
            batch.bind(&mut shared, reservation, &missing),
            Err(Failure::BadGeometry)
        );
        batch
            .bind(&mut shared, reservation, &bindings(&tickets))
            .unwrap();
        let before = batch.ranges(tickets[0]).unwrap().to_vec();
        let mut changed = bindings(&tickets);
        changed[0].1[0].span.address += 1;
        assert_eq!(
            batch.bind(&mut shared, reservation, &changed),
            Err(Failure::AlreadyOwned)
        );
        assert_eq!(batch.ranges(tickets[0]).unwrap(), before);
    }

    #[test]
    fn wrong_slot_session_cursor_width_and_generation_cannot_commit_or_erase_owner() {
        for field in 0..5 {
            let (mut shared, mut batch, reservation, tickets) = bound(2);
            batch
                .begin_frame(&mut shared, reservation, &rows(&tickets))
                .unwrap();
            let mut foreign = tickets[0];
            match field {
                0 => foreign.ticket.owner.slot += 1,
                1 => foreign.ticket.owner.session += 1,
                2 => foreign.ticket.owner.cursor += 1,
                3 => foreign.ticket.owner.rows -= 1,
                _ => foreign.ticket.generation += 1,
            }
            assert_eq!(
                batch.commit_succeeded(&mut shared, foreign, 3),
                Err(Failure::StaleOwner)
            );
            assert!(batch.owns_for_cleanup(&shared, tickets[0]));
            assert!(!batch.owns_for_cleanup(&shared, foreign));
            assert_eq!(shared.check(), Err(Failure::Poisoned));
        }
    }

    #[test]
    fn failed_commit_retains_ticket_and_poisons_uncertain_retry() {
        let (mut shared, mut batch, reservation, tickets) = bound(2);
        batch
            .begin_frame(&mut shared, reservation, &rows(&tickets))
            .unwrap();
        assert_eq!(
            batch.commit_failed(&mut shared, tickets[0]),
            Err(Failure::Poisoned)
        );
        assert_eq!(batch.state(tickets[0]), Some(LaneState::InFlight));
        assert!(batch.owns_for_cleanup(&shared, tickets[0]));
        assert_eq!(
            batch.commit_succeeded(&mut shared, tickets[0], 2),
            Err(Failure::Poisoned)
        );
        assert_eq!(
            batch.finish(&mut shared, reservation),
            Err(Failure::Poisoned)
        );
    }

    #[test]
    fn unpublished_hidden_keeps_the_reservation_until_every_publication() {
        let (mut shared, mut batch, reservation, tickets) = bound(2);
        batch
            .begin_frame(&mut shared, reservation, &rows(&tickets))
            .unwrap();
        batch.commit_succeeded(&mut shared, tickets[0], 2).unwrap();
        batch.commit_succeeded(&mut shared, tickets[1], 5).unwrap();
        batch.publish_hidden(&mut shared, tickets[1], 5).unwrap();
        assert_eq!(
            batch.end_frame(&mut shared, reservation),
            Err(Failure::AlreadyOwned)
        );
        assert!(batch.owns_for_cleanup(&shared, tickets[0]));
        batch.publish_hidden(&mut shared, tickets[0], 2).unwrap();
        batch.end_frame(&mut shared, reservation).unwrap();
        batch.finish(&mut shared, reservation).unwrap();
    }

    #[test]
    fn every_nonempty_k5_prefix_is_owned_and_invalid_prefixes_poison() {
        for accepted in 1..=K {
            let (mut shared, mut batch, reservation, tickets) = bound(2);
            batch
                .begin_frame(&mut shared, reservation, &rows(&tickets))
                .unwrap();
            for ticket in &tickets {
                batch
                    .commit_succeeded(&mut shared, *ticket, accepted)
                    .unwrap();
                batch
                    .publish_hidden(&mut shared, *ticket, accepted)
                    .unwrap();
            }
            batch.end_frame(&mut shared, reservation).unwrap();
            batch.finish(&mut shared, reservation).unwrap();
        }
        for accepted in [0, K + 1, usize::MAX] {
            let (mut shared, mut batch, reservation, tickets) = bound(2);
            batch
                .begin_frame(&mut shared, reservation, &rows(&tickets))
                .unwrap();
            assert_eq!(
                batch.commit_succeeded(&mut shared, tickets[0], accepted),
                Err(Failure::BadGeometry)
            );
            assert!(batch.owns_for_cleanup(&shared, tickets[0]));
            assert_eq!(shared.check(), Err(Failure::Poisoned));
        }
    }

    #[test]
    fn publication_requires_exact_accepted_prefix_and_is_single_use() {
        for wrong in [0, 1, 4, 5] {
            let (mut shared, mut batch, reservation, tickets) = bound(2);
            batch
                .begin_frame(&mut shared, reservation, &rows(&tickets))
                .unwrap();
            batch.commit_succeeded(&mut shared, tickets[0], 3).unwrap();
            assert_eq!(
                batch.publish_hidden(&mut shared, tickets[0], wrong),
                Err(Failure::StaleOwner)
            );
            assert!(batch.owns_for_cleanup(&shared, tickets[0]));
        }
        let (mut shared, mut batch, reservation, tickets) = bound(2);
        batch
            .begin_frame(&mut shared, reservation, &rows(&tickets))
            .unwrap();
        publish(&mut batch, &mut shared, &tickets);
        assert_eq!(
            batch.publish_hidden(&mut shared, tickets[0], 3),
            Err(Failure::StaleOwner)
        );
    }

    #[test]
    fn partial_effect_failure_poisons_and_every_lane_cleanup_is_still_attempted() {
        let (mut shared, mut batch, reservation, tickets) = bound(8);
        batch
            .begin_frame(&mut shared, reservation, &rows(&tickets[..4]))
            .unwrap();
        batch.commit_succeeded(&mut shared, tickets[0], 4).unwrap();
        batch.publish_hidden(&mut shared, tickets[0], 4).unwrap();
        assert_eq!(
            batch.fail_execution::<()>(&mut shared, reservation),
            Err(Failure::Poisoned)
        );
        let mut attempted = Vec::new();
        let result = batch.cleanup_all(&mut shared, reservation, |ticket, _state| {
            attempted.push(ticket.owner().slot);
            if ticket.owner().slot == 1 {
                Err(())
            } else {
                Ok(())
            }
        });
        assert_eq!(result, Err(Failure::Poisoned));
        assert_eq!(attempted, (0..8).collect::<Vec<_>>());
        assert_eq!(batch.state(tickets[1]), Some(LaneState::InFlight));
        assert!(batch.owns_for_cleanup(&shared, tickets[1]));
        assert_eq!(
            batch.finish(&mut shared, reservation),
            Err(Failure::Poisoned)
        );
        assert_eq!(shared.require_idle(), Err(Failure::Poisoned));
    }

    #[test]
    fn cancelled_cohort_can_abort_every_lane_without_releasing_early() {
        let (mut shared, mut batch, reservation, tickets) = bound(2);
        batch
            .begin_frame(&mut shared, reservation, &rows(&tickets))
            .unwrap();
        let mut attempts = 0;
        batch
            .cleanup_all(&mut shared, reservation, |_, _| {
                attempts += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(attempts, 2);
        assert_eq!(shared.require_idle(), Err(Failure::AlreadyOwned));
        batch.end_frame(&mut shared, reservation).unwrap();
        batch.finish(&mut shared, reservation).unwrap();
        assert!(shared.require_idle().is_ok());
    }

    #[test]
    fn rejected_shapes_and_busy_admission_leave_scalar_c1_semantics_intact() {
        let mut shared = Transactions::default();
        let scalar = shared.begin(owner(0)).unwrap();
        let mut batch = BatchTransactions::default();
        assert_eq!(
            batch.reserve(&mut shared, &owners(2), 8),
            Err(Failure::AlreadyOwned)
        );
        assert_eq!(shared.check(), Ok(()));
        shared.commit_failed(scalar).unwrap();
        shared.commit_succeeded(scalar, 3).unwrap();
        for bad in [owners(1), owners(9), vec![owner(0), owner(0)]] {
            assert_eq!(
                batch.reserve(&mut shared, &bad, 8),
                Err(Failure::BadGeometry)
            );
            assert_eq!(shared.require_idle(), Ok(()));
        }
        let (reservation, _) = batch.reserve(&mut shared, &owners(2), 8).unwrap();
        assert_eq!(shared.require_idle(), Err(Failure::AlreadyOwned));
        assert_eq!(
            batch.reserve(&mut shared, &owners(2), 8),
            Err(Failure::AlreadyOwned)
        );
        assert!(shared.owns_for_cleanup(reservation.0));
    }

    #[test]
    fn released_slot_session_and_cursor_reuse_gets_a_fresh_generation() {
        let (mut shared, mut batch, old, tickets) = bound(2);
        batch
            .begin_frame(&mut shared, old, &rows(&tickets))
            .unwrap();
        publish(&mut batch, &mut shared, &tickets);
        batch.end_frame(&mut shared, old).unwrap();
        batch.finish(&mut shared, old).unwrap();
        let (current, next) = batch.reserve(&mut shared, &owners(2), 8).unwrap();
        assert_ne!(old, current);
        assert_ne!(tickets[0], next[0]);
        assert_eq!(
            batch.commit_succeeded(&mut shared, tickets[0], 2),
            Err(Failure::StaleOwner)
        );
        assert!(batch.owns_for_cleanup(&shared, next[0]));
        assert!(!batch.owns_for_cleanup(&shared, tickets[0]));
    }

    #[test]
    fn c52_cannot_be_flattened_into_a_260_row_forward() {
        let (mut shared, mut batch, reservation, tickets) = bound(52);
        assert_eq!(
            batch.begin_frame(&mut shared, reservation, &rows(&tickets)),
            Err(Failure::BadGeometry)
        );
        assert!(
            tickets
                .iter()
                .all(|ticket| batch.state(*ticket) == Some(LaneState::Bound))
        );
        batch
            .begin_frame(&mut shared, reservation, &rows(&tickets[..32]))
            .unwrap();
        assert_eq!(
            batch.begin_frame(&mut shared, reservation, &rows(&tickets[32..])),
            Err(Failure::AlreadyOwned)
        );
    }

    #[test]
    fn terminal_commits_retire_without_fabricating_hidden_publications() {
        let (mut shared, mut batch, reservation, tickets) = bound(2);
        batch
            .begin_frame(&mut shared, reservation, &rows(&tickets))
            .unwrap();
        batch.commit_succeeded(&mut shared, tickets[0], 1).unwrap();
        batch.retire_committed(&mut shared, tickets[0]).unwrap();
        batch.abort_succeeded(&mut shared, tickets[1]).unwrap();
        assert_eq!(
            batch.state(tickets[0]),
            Some(LaneState::Retired { accepted: 1 })
        );
        assert_eq!(batch.frame_row_base(tickets[0]), Some(0));
        assert_eq!(batch.frame_row_base(tickets[1]), Some(K));
        batch.end_frame(&mut shared, reservation).unwrap();
        assert_eq!(batch.frame_row_base(tickets[0]), None);
        batch.finish(&mut shared, reservation).unwrap();
        assert!(shared.require_idle().is_ok());
    }

    #[test]
    fn terminal_retirement_requires_a_successful_positive_commit() {
        let (mut shared, mut batch, reservation, tickets) = bound(2);
        batch
            .begin_frame(&mut shared, reservation, &rows(&tickets))
            .unwrap();
        assert_eq!(
            batch.retire_committed(&mut shared, tickets[0]),
            Err(Failure::StaleOwner)
        );
        assert!(batch.owns_for_cleanup(&shared, tickets[0]));
        assert_eq!(shared.require_idle(), Err(Failure::Poisoned));
    }

    #[test]
    fn preparation_failure_can_cancel_all_reserved_lanes_without_a_frame() {
        let (mut shared, mut batch, reservation, tickets) = reserve(8);
        let mut attempted = Vec::new();
        batch
            .cleanup_all(&mut shared, reservation, |ticket, state| {
                assert_eq!(state, LaneState::Reserved);
                attempted.push(ticket.owner().slot);
                Ok(())
            })
            .unwrap();
        assert_eq!(attempted, (0..8).collect::<Vec<_>>());
        assert!(
            tickets
                .iter()
                .all(|ticket| batch.state(*ticket) == Some(LaneState::Aborted))
        );
        batch.finish_cancelled(&mut shared, reservation).unwrap();
        assert!(shared.require_idle().is_ok());
    }

    fn owners_k(count: usize, k: usize) -> Vec<Owner> {
        (0..count)
            .map(|slot| Owner {
                rows: k,
                ..owner(slot)
            })
            .collect()
    }
    fn rows_k(tickets: &[LaneTicket], k: usize) -> Vec<FrameRow> {
        tickets
            .iter()
            .enumerate()
            .map(|(i, lane)| FrameRow {
                lane: *lane,
                row_base: i * k,
            })
            .collect()
    }

    fn owners_ragged(ks: &[usize]) -> Vec<Owner> {
        ks.iter()
            .enumerate()
            .map(|(slot, &k)| Owner {
                rows: k,
                ..owner(slot)
            })
            .collect()
    }

    fn rows_ragged(tickets: &[LaneTicket]) -> Vec<FrameRow> {
        let mut base = 0usize;
        tickets
            .iter()
            .map(|lane| {
                let row = FrameRow {
                    lane: *lane,
                    row_base: base,
                };
                base += lane.owner().rows;
                row
            })
            .collect()
    }

    #[test]
    fn fixed_mode_admits_only_k5_and_adaptive_admits_k1_through_k8() {
        for k in 0..=12 {
            assert_eq!(width_admitted(k, false), k == 5, "k={k}");
            assert_eq!(width_admitted(k, true), (1..=8).contains(&k), "k={k}");
        }
    }

    #[test]
    fn adaptive_widths_own_one_full_frame_and_every_prefix() {
        // Representative (seqs, k) pairs well within the 256-row / 64-seq frame ceiling.
        for (seqs, k) in [
            (16, 8),
            (18, 7),
            (21, 6),
            (25, 5),
            (32, 4),
            (42, 3),
            (64, 2),
        ] {
            for accepted in 1..=k {
                let mut shared = Transactions::default();
                let mut batch = BatchTransactions::default();
                let (reservation, tickets) = batch
                    .reserve(&mut shared, &owners_k(seqs, k), MAX_LANES)
                    .unwrap();
                batch
                    .bind(&mut shared, reservation, &bindings(&tickets))
                    .unwrap();
                batch
                    .begin_frame(&mut shared, reservation, &rows_k(&tickets, k))
                    .unwrap();
                for (s, ticket) in tickets.iter().enumerate() {
                    assert_eq!(batch.frame_row_base(*ticket), Some(s * k));
                    batch
                        .commit_succeeded(&mut shared, *ticket, accepted)
                        .unwrap();
                    batch
                        .publish_hidden(&mut shared, *ticket, accepted)
                        .unwrap();
                }
                batch.end_frame(&mut shared, reservation).unwrap();
                batch.finish(&mut shared, reservation).unwrap();
                assert!(shared.require_idle().is_ok(), "seqs={seqs} k={k}");
            }
            let mut shared = Transactions::default();
            let mut batch = BatchTransactions::default();
            let (reservation, tickets) = batch
                .reserve(&mut shared, &owners_k(seqs, k), MAX_LANES)
                .unwrap();
            batch
                .bind(&mut shared, reservation, &bindings(&tickets))
                .unwrap();
            batch
                .begin_frame(&mut shared, reservation, &rows_k(&tickets, k))
                .unwrap();
            assert_eq!(
                batch.commit_succeeded(&mut shared, tickets[0], k + 1),
                Err(Failure::BadGeometry)
            );
            assert_eq!(shared.check(), Err(Failure::Poisoned));
        }
    }

    #[test]
    fn cohorts_refuse_out_of_range_widths_and_overfull_frames() {
        let mut shared = Transactions::default();
        let mut batch = BatchTransactions::default();
        for bad in [owners_k(2, 0), owners_k(2, 9), owners_k(2, 10)] {
            assert_eq!(
                batch.reserve(&mut shared, &bad, MAX_LANES),
                Err(Failure::BadGeometry)
            );
        }
        assert!(shared.require_idle().is_ok(), "shape refusals never poison");
        // 33 K8 owners (264 rows) fit the lanes but not one 256-row frame.
        {
            let mut shared_k8 = Transactions::default();
            let mut batch_k8 = BatchTransactions::default();
            let (res_k8, tickets_k8) = batch_k8
                .reserve(&mut shared_k8, &owners_k(33, 8), MAX_LANES)
                .unwrap();
            batch_k8
                .bind(&mut shared_k8, res_k8, &bindings(&tickets_k8))
                .unwrap();
            assert_eq!(
                batch_k8.begin_frame(&mut shared_k8, res_k8, &rows_k(&tickets_k8, 8)),
                Err(Failure::BadGeometry)
            );
            batch_k8
                .begin_frame(&mut shared_k8, res_k8, &rows_k(&tickets_k8[..32], 8))
                .unwrap();
        }
        // 65 K4 owners (260 rows) fit the lanes but not one 256-row / 64-seq frame.
        let (reservation, tickets) = batch
            .reserve(&mut shared, &owners_k(65, 4), MAX_LANES)
            .unwrap();
        batch
            .bind(&mut shared, reservation, &bindings(&tickets))
            .unwrap();
        assert_eq!(
            batch.begin_frame(&mut shared, reservation, &rows_k(&tickets, 4)),
            Err(Failure::BadGeometry)
        );
        // A K5 row map cannot address a K4 cohort.
        assert_eq!(
            batch.begin_frame(&mut shared, reservation, &rows_k(&tickets[..2], 5)),
            Err(Failure::BadGeometry)
        );
        batch
            .begin_frame(&mut shared, reservation, &rows_k(&tickets[..64], 4))
            .unwrap();
    }

    #[test]
    fn ragged_mixed_k1_through_k8_cohort_tracks_cumulative_row_bases_and_per_lane_commits() {
        let ks = [7, 3, 1, 5, 8, 2, 6, 4];
        let mut shared = Transactions::default();
        let mut batch = BatchTransactions::default();
        let (reservation, tickets) = batch
            .reserve(&mut shared, &owners_ragged(&ks), MAX_LANES)
            .unwrap();
        assert_eq!(batch.cohort_width_bounds(), Some((1, 8)));
        batch
            .bind(&mut shared, reservation, &bindings(&tickets))
            .unwrap();

        // Uniform row_base map is rejected for ragged widths without mutating lanes.
        assert_eq!(
            batch.begin_frame(&mut shared, reservation, &rows_k(&tickets, 8)),
            Err(Failure::BadGeometry)
        );

        let frame_rows = rows_ragged(&tickets);
        batch
            .begin_frame(&mut shared, reservation, &frame_rows)
            .unwrap();

        let mut expected_base = 0usize;
        for (i, &k_s) in ks.iter().enumerate() {
            assert_eq!(batch.frame_row_base(tickets[i]), Some(expected_base));
            expected_base += k_s;
            let accepted = k_s;
            batch
                .commit_succeeded(&mut shared, tickets[i], accepted)
                .unwrap();
            batch
                .publish_hidden(&mut shared, tickets[i], accepted)
                .unwrap();
        }
        batch.end_frame(&mut shared, reservation).unwrap();
        batch.finish(&mut shared, reservation).unwrap();
        assert!(shared.require_idle().is_ok());

        // K=1 anchor-only lane rejects accepted == 2 and poisons.
        let mut shared_k1 = Transactions::default();
        let mut batch_k1 = BatchTransactions::default();
        let (res_k1, tickets_k1) = batch_k1
            .reserve(&mut shared_k1, &owners_ragged(&[1, 7]), MAX_LANES)
            .unwrap();
        batch_k1
            .bind(&mut shared_k1, res_k1, &bindings(&tickets_k1))
            .unwrap();
        batch_k1
            .begin_frame(&mut shared_k1, res_k1, &rows_ragged(&tickets_k1))
            .unwrap();
        assert_eq!(batch_k1.frame_row_base(tickets_k1[0]), Some(0));
        assert_eq!(batch_k1.frame_row_base(tickets_k1[1]), Some(1));
        assert_eq!(
            batch_k1.commit_succeeded(&mut shared_k1, tickets_k1[0], 2),
            Err(Failure::BadGeometry)
        );
        assert_eq!(shared_k1.check(), Err(Failure::Poisoned));

        // 65 K=1 lanes exceed MAX_FRAME_SEQS (64) even though 65 <= MAX_FRAME_ROWS (256).
        let mut shared_65 = Transactions::default();
        let mut batch_65 = BatchTransactions::default();
        let (res_65, tickets_65) = batch_65
            .reserve(&mut shared_65, &owners_k(65, 1), MAX_LANES)
            .unwrap();
        batch_65
            .bind(&mut shared_65, res_65, &bindings(&tickets_65))
            .unwrap();
        assert_eq!(
            batch_65.begin_frame(&mut shared_65, res_65, &rows_ragged(&tickets_65)),
            Err(Failure::BadGeometry)
        );
        batch_65
            .begin_frame(&mut shared_65, res_65, &rows_ragged(&tickets_65[..64]))
            .unwrap();
    }

    #[test]
    fn lanes_cover_the_c128_lane_and_c64_adaptive_ceiling() {
        let mut shared = Transactions::default();
        let mut batch = BatchTransactions::default();
        let (reservation, _) = batch.reserve(&mut shared, &owners_k(64, 2), 64).unwrap();
        assert!(batch.reservation(&shared, reservation));
        let mut shared = Transactions::default();
        let mut batch = BatchTransactions::default();
        let (reservation_max, _) = batch.reserve(&mut shared, &owners_k(128, 2), 128).unwrap();
        assert!(batch.reservation(&shared, reservation_max));
        let mut shared = Transactions::default();
        let mut batch = BatchTransactions::default();
        let beyond = [
            Owner {
                rows: 2,
                ..owner(0)
            },
            Owner {
                rows: 2,
                ..owner(64)
            },
        ];
        assert_eq!(
            batch.reserve(&mut shared, &beyond, 64),
            Err(Failure::BadGeometry)
        );
        assert_eq!(
            batch.reserve(&mut shared, &owners_k(129, 2), 129),
            Err(Failure::BadGeometry)
        );
        assert_eq!(
            batch.reserve(&mut shared, &owners_k(2, 2), MAX_LANES + 1),
            Err(Failure::BadGeometry)
        );
    }
}
