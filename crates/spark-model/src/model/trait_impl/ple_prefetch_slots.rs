// SPDX-License-Identifier: AGPL-3.0-only

//! Routing of the two chunk-0 PLE reads that live outside a sequence slot in
//! the model's prefetch map:
//!
//! * the **admission** read (`ATLAS_QWEN4_PLE_EARLY_ISSUE`), started when the
//!   scheduler admits a request and consumed by its chunk 0;
//! * the **queue** read (`ATLAS_QWEN4_PLE_QUEUE_PREFETCH`), started for the
//!   pending-queue head while another request runs.
//!
//! Invariants (tested below):
//! 1. A read is consumed only by a chunk 0 whose exact tokens it was issued
//!    for: every hand-off compares the full token key.
//! 2. At most one queue read exists; a new head replaces it (dropped, not
//!    joined: the yielding read stops at the block it is on). Each read holds
//!    at most one chunk's rows.
//! 3. A queue read never displaces the admission read, and admission adopts
//!    a queue read only for the same tokens, so a cancelled or reordered
//!    queue head can never hand its rows to another request.

use std::collections::HashMap;

/// Prefetch-map key of the admission read. Sequence slots are
/// `< max_batch_size`, so neither key names a real slot.
pub(super) const ADMISSION_SLOT: usize = usize::MAX;
/// Prefetch-map key of the (single) queue read.
pub(super) const QUEUE_SLOT: usize = usize::MAX - 1;

/// What a map entry must answer: was it issued for exactly this chunk 0?
pub(super) trait ChunkZeroRead {
    fn is_for(&self, current: &[u32]) -> bool;
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
impl ChunkZeroRead for crate::layers::qwen4_ple::PleChunkPrefetch {
    fn is_for(&self, current: &[u32]) -> bool {
        self.matches(0, current, &[])
    }
}

/// Outcome of routing an admission for `current`.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum AdmissionRoute {
    /// The admission slot already holds this read: nothing to start.
    AlreadyIssued,
    /// The queue read for these tokens moved into the admission slot.
    AdoptedQueued,
    /// Start a fresh admission read. `discarded` = a stale admission read
    /// for other tokens was dropped first.
    Start { discarded: bool },
}

/// Admission for `current`: keep a matching admission read, else adopt a
/// matching queue read, else drop any stale admission read and start one.
/// A queue read for other tokens is left alone (its request is still pending).
pub(super) fn route_admission<T: ChunkZeroRead>(
    map: &mut HashMap<usize, T>,
    current: &[u32],
) -> AdmissionRoute {
    if map
        .get(&ADMISSION_SLOT)
        .is_some_and(|read| read.is_for(current))
    {
        return AdmissionRoute::AlreadyIssued;
    }
    if map
        .get(&QUEUE_SLOT)
        .is_some_and(|read| read.is_for(current))
    {
        let queued = map.remove(&QUEUE_SLOT).expect("checked");
        map.insert(ADMISSION_SLOT, queued);
        return AdmissionRoute::AdoptedQueued;
    }
    AdmissionRoute::Start {
        discarded: map.remove(&ADMISSION_SLOT).is_some(),
    }
}

/// Outcome of routing a queue hand-over for `current`.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum QueueRoute {
    /// A read for these tokens already exists (queued, or already admitted).
    AlreadyIssued,
    /// Start a queue read. `replaced` = the previous head's queue read was
    /// dropped first (the head changed, or that request was cancelled).
    Start { replaced: bool },
}

pub(super) fn route_queue<T: ChunkZeroRead>(
    map: &mut HashMap<usize, T>,
    current: &[u32],
) -> QueueRoute {
    let issued = |slot| map.get(&slot).is_some_and(|read: &T| read.is_for(current));
    if issued(QUEUE_SLOT) || issued(ADMISSION_SLOT) {
        return QueueRoute::AlreadyIssued;
    }
    QueueRoute::Start {
        replaced: map.remove(&QUEUE_SLOT).is_some(),
    }
}

/// Chunk 0's hand-off: the admission read if it was issued for exactly
/// `current`; a read for other tokens is returned as `Err` for the caller to
/// discard (it is never consumed).
pub(super) fn take_for_chunk_zero<T: ChunkZeroRead>(
    map: &mut HashMap<usize, T>,
    current: &[u32],
) -> Option<Result<T, T>> {
    let read = map.remove(&ADMISSION_SLOT)?;
    Some(if read.is_for(current) {
        Ok(read)
    } else {
        Err(read)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A read tagged with the tokens it was issued for.
    #[derive(Debug, PartialEq, Eq)]
    struct Read(Vec<u32>);

    impl ChunkZeroRead for Read {
        fn is_for(&self, current: &[u32]) -> bool {
            self.0 == current
        }
    }

    fn queue(map: &mut HashMap<usize, Read>, tokens: &[u32]) -> QueueRoute {
        let route = route_queue(map, tokens);
        if matches!(route, QueueRoute::Start { .. }) {
            map.insert(QUEUE_SLOT, Read(tokens.to_vec()));
        }
        route
    }

    fn admit(map: &mut HashMap<usize, Read>, tokens: &[u32]) -> AdmissionRoute {
        let route = route_admission(map, tokens);
        if matches!(route, AdmissionRoute::Start { .. }) {
            map.insert(ADMISSION_SLOT, Read(tokens.to_vec()));
        }
        route
    }

    /// The rows chunk 0 would use: the prefetched read only if it is its own.
    fn chunk_zero(map: &mut HashMap<usize, Read>, tokens: &[u32]) -> Option<Read> {
        take_for_chunk_zero(map, tokens).and_then(Result::ok)
    }

    #[test]
    fn queued_head_is_adopted_by_its_own_admission_and_consumed_once() {
        let mut map = HashMap::new();
        let a = [1, 2, 3];
        assert_eq!(queue(&mut map, &a), QueueRoute::Start { replaced: false });
        // Every later tick for the same head: no second read.
        assert_eq!(queue(&mut map, &a), QueueRoute::AlreadyIssued);
        assert_eq!(admit(&mut map, &a), AdmissionRoute::AdoptedQueued);
        assert!(!map.contains_key(&QUEUE_SLOT), "adopted, not copied");
        assert_eq!(chunk_zero(&mut map, &a), Some(Read(a.to_vec())));
        assert!(map.is_empty());
    }

    #[test]
    fn a_cancelled_or_replaced_head_never_feeds_another_request() {
        let mut map = HashMap::new();
        let (a, b) = ([1, 2, 3], [9, 9, 9]);
        // A is queued, then cancelled; B becomes the head.
        queue(&mut map, &a);
        assert_eq!(queue(&mut map, &b), QueueRoute::Start { replaced: true });
        assert_eq!(map.len(), 1, "one queue read at a time");
        assert_eq!(admit(&mut map, &b), AdmissionRoute::AdoptedQueued);
        assert_eq!(chunk_zero(&mut map, &b), Some(Read(b.to_vec())));
        // B arrives again with A's rows nowhere in the map.
        assert!(map.is_empty());
    }

    #[test]
    fn a_reordered_admission_leaves_the_queue_read_to_its_owner() {
        let mut map = HashMap::new();
        let (a, c) = ([1, 2, 3], [5, 6]);
        queue(&mut map, &a);
        // C is admitted first (e.g. it was already drained): it reads its own.
        assert_eq!(
            admit(&mut map, &c),
            AdmissionRoute::Start { discarded: false }
        );
        assert_eq!(chunk_zero(&mut map, &c), Some(Read(c.to_vec())));
        // A's queue read survived and is adopted by A.
        assert_eq!(admit(&mut map, &a), AdmissionRoute::AdoptedQueued);
        assert_eq!(chunk_zero(&mut map, &a), Some(Read(a.to_vec())));
    }

    #[test]
    fn a_queue_read_never_displaces_the_admission_read() {
        let mut map = HashMap::new();
        let (a, b) = ([1, 2, 3], [4]);
        admit(&mut map, &a);
        // B is the next head while A is still being admitted.
        assert_eq!(queue(&mut map, &b), QueueRoute::Start { replaced: false });
        assert_eq!(map.get(&ADMISSION_SLOT), Some(&Read(a.to_vec())));
        // A head that is already admitted is not read twice.
        assert_eq!(queue(&mut map, &a), QueueRoute::AlreadyIssued);
    }

    #[test]
    fn a_stale_admission_read_is_dropped_and_never_consumed() {
        let mut map = HashMap::new();
        let (a, b) = ([1, 2, 3], [7, 8]);
        admit(&mut map, &a); // A then failed before chunk 0
        assert_eq!(
            admit(&mut map, &b),
            AdmissionRoute::Start { discarded: true }
        );
        assert_eq!(chunk_zero(&mut map, &b), Some(Read(b.to_vec())));
        // A chunk 0 whose tokens differ gets the read back as Err to discard.
        admit(&mut map, &a);
        assert_eq!(
            take_for_chunk_zero(&mut map, &b),
            Some(Err(Read(a.to_vec())))
        );
        assert!(map.is_empty());
    }
}
