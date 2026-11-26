// SPDX-License-Identifier: AGPL-3.0-only

//! Absolute and resident state transitions for the bounded DFlash ring.

use super::{Result, RingCopyPlan, RingPlanError, RingSpanPlan, plan_ring_copy, plan_write_chunk};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RingState {
    pub max_context: usize,
    pub capacity: usize,
    pub absolute_len: usize,
    pub resident_len: usize,
    /// Lowest absolute token position whose target hidden state was actually
    /// captured into the ring. Zero on uncached prefills; advanced to `M` on a
    /// Marconi prefix-cache skip of `[0..M)` so downstream cross-attention
    /// never reads uninitialized slots `< M`.
    pub ctx_valid_from_abs: usize,
}

impl RingState {
    pub(crate) fn new(max_context: usize, capacity: usize) -> Result<Self> {
        Self::from_lengths(max_context, capacity, 0, 0)
    }

    pub(crate) fn from_lengths(
        max_context: usize,
        capacity: usize,
        absolute_len: usize,
        resident_len: usize,
    ) -> Result<Self> {
        Self::from_lengths_with_valid_from(max_context, capacity, absolute_len, resident_len, 0)
    }

    pub(crate) fn from_lengths_with_valid_from(
        max_context: usize,
        capacity: usize,
        absolute_len: usize,
        resident_len: usize,
        ctx_valid_from_abs: usize,
    ) -> Result<Self> {
        if max_context == 0 || capacity == 0 || capacity > max_context {
            return Err(RingPlanError("ring capacity is incompatible with context"));
        }
        if ctx_valid_from_abs > absolute_len || absolute_len > max_context {
            return Err(RingPlanError(
                "absolute and valid-from lengths are inconsistent",
            ));
        }
        let expected_resident = (absolute_len - ctx_valid_from_abs).min(capacity);
        if resident_len != expected_resident {
            return Err(RingPlanError(
                "absolute and resident lengths are inconsistent",
            ));
        }
        Ok(Self {
            max_context,
            capacity,
            absolute_len,
            resident_len,
            ctx_valid_from_abs,
        })
    }

    /// Advance the absolute cursor past a Marconi prefix-cache hit `[0..skip_to)`
    /// whose target hidden states were never computed. Sets
    /// `absolute_len = skip_to`, `resident_len = 0`, and
    /// `ctx_valid_from_abs = skip_to` so `resident_start()` stays `>= skip_to`
    /// while `ctx_len` remains aligned with `seq.seq_len`.
    pub(crate) fn advance_skipped_prefix(&self, skip_to: usize) -> Result<Self> {
        if skip_to < self.absolute_len || skip_to > self.max_context {
            return Err(RingPlanError(
                "skipped prefix cursor is out of bounds for ring state",
            ));
        }
        if skip_to == self.absolute_len {
            return Ok(*self);
        }
        Self::from_lengths_with_valid_from(self.max_context, self.capacity, skip_to, 0, skip_to)
    }

    pub(crate) fn resident_start(&self) -> usize {
        self.absolute_len - self.resident_len
    }

    pub(crate) fn slot_for(&self, absolute_position: usize) -> Result<usize> {
        if absolute_position < self.resident_start() || absolute_position >= self.absolute_len {
            return Err(RingPlanError("absolute position is not resident"));
        }
        Ok(absolute_position % self.capacity)
    }

    /// Gather a fully resident logical range in chronological order.
    pub(crate) fn plan_gather(&self, absolute_start: usize, rows: usize) -> Result<RingCopyPlan> {
        if rows == 0 {
            return Err(RingPlanError("gather row count must be non-zero"));
        }
        let absolute_end = absolute_start
            .checked_add(rows)
            .ok_or(RingPlanError("absolute gather end overflowed"))?;
        if absolute_start < self.resident_start() || absolute_end > self.absolute_len {
            return Err(RingPlanError("gather range is not fully resident"));
        }
        plan_ring_copy(
            self.resident_start(),
            self.absolute_len,
            self.capacity,
            absolute_start,
            absolute_end,
        )
    }

    /// Plan bootstrap plus accepted target rows and advance absolute context.
    pub(crate) fn plan_accepted_append(&self, rows: usize) -> Result<AppendPlan> {
        let write = plan_write_chunk(self.absolute_len, rows, self.max_context, self.capacity)?;
        let absolute_len = self
            .absolute_len
            .checked_add(rows)
            .ok_or(RingPlanError("absolute append end overflowed"))?;
        let resident_len = (absolute_len - self.ctx_valid_from_abs).min(self.capacity);
        let next = Self::from_lengths_with_valid_from(
            self.max_context,
            self.capacity,
            absolute_len,
            resident_len,
            self.ctx_valid_from_abs,
        )?;
        Ok(AppendPlan { write, next })
    }

    /// The state after a prefill chunk of `rows` rows at `absolute_start`,
    /// or `None` when the cursor is already there. Chunked prefill advances
    /// after every chunk (the next chunk's capture plans against this cursor)
    /// and the last-chunk finalizer repeats the call for the same rows.
    pub(crate) fn advanced_after_chunk(
        &self,
        absolute_start: usize,
        rows: usize,
    ) -> Result<Option<RingState>> {
        let end = absolute_start
            .checked_add(rows)
            .ok_or(RingPlanError("absolute chunk end overflowed"))?;
        if rows == 0 || self.absolute_len == end {
            return Ok(None);
        }
        Ok(Some(self.plan_append_at(absolute_start, rows)?.next))
    }

    /// Plan an append only when the caller's absolute cursor is exact.
    pub(crate) fn plan_append_at(&self, absolute_start: usize, rows: usize) -> Result<AppendPlan> {
        if absolute_start != self.absolute_len {
            return Err(RingPlanError("append start does not match absolute cursor"));
        }
        self.plan_accepted_append(rows)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AppendPlan {
    pub write: RingSpanPlan,
    pub next: RingState,
}

#[cfg(test)]
mod marconi_valid_from_tests {
    use super::RingState;

    #[test]
    fn marconi_skip_preserves_ctx_len_alignment_and_forbids_reads_below_valid_from() {
        let ring = RingState::new(16_384, 4_096).unwrap();

        // Simulate a 2-chunk Marconi prefix hit covering [0..1024), where chunk 0
        // early-returns at 512 and chunk 1 recomputes only the final token at 1023.
        let after_chunk0 = ring.advance_skipped_prefix(512).unwrap();
        assert_eq!(after_chunk0.absolute_len, 512);
        assert_eq!(after_chunk0.resident_len, 0);
        assert_eq!(after_chunk0.ctx_valid_from_abs, 512);
        assert_eq!(after_chunk0.resident_start(), 512);

        let before_last_tok = after_chunk0.advance_skipped_prefix(1023).unwrap();
        assert_eq!(before_last_tok.absolute_len, 1023);
        assert_eq!(before_last_tok.resident_len, 0);
        assert_eq!(before_last_tok.ctx_valid_from_abs, 1023);
        assert!(before_last_tok.slot_for(0).is_err());
        assert!(before_last_tok.slot_for(1022).is_err());

        // Capture the recomputed final prompt token at position 1023.
        let after_prefill = before_last_tok
            .advanced_after_chunk(1023, 1)
            .unwrap()
            .unwrap();
        assert_eq!(after_prefill.absolute_len, 1024);
        assert_eq!(after_prefill.resident_len, 1);
        assert_eq!(after_prefill.ctx_valid_from_abs, 1023);
        assert_eq!(after_prefill.resident_start(), 1023);
        assert!(after_prefill.slot_for(1022).is_err());
        assert_eq!(after_prefill.slot_for(1023).unwrap(), 1023);
        assert!(after_prefill.plan_gather(1022, 2).is_err());
        assert!(after_prefill.plan_gather(1023, 1).is_ok());

        // Subsequent decode appends grow resident_len from ctx_valid_from_abs and
        // cap at ring capacity once absolute_len - ctx_valid_from_abs > capacity.
        let after_decode = after_prefill.plan_append_at(1024, 5).unwrap().next;
        assert_eq!(after_decode.absolute_len, 1029);
        assert_eq!(after_decode.resident_len, 6);
        assert_eq!(after_decode.ctx_valid_from_abs, 1023);
        assert_eq!(after_decode.resident_start(), 1023);
        assert!(after_decode.plan_gather(1022, 1).is_err());
        assert!(after_decode.plan_gather(1023, 6).is_ok());

        // Advancing beyond capacity moves resident_start strictly above valid_from.
        let wrapped = after_decode.plan_append_at(1029, 4_100).unwrap().next;
        assert_eq!(wrapped.absolute_len, 5_129);
        assert_eq!(wrapped.resident_len, 4_096);
        assert_eq!(wrapped.ctx_valid_from_abs, 1023);
        assert_eq!(wrapped.resident_start(), 1_033);
        assert!(wrapped.resident_start() > wrapped.ctx_valid_from_abs);
        assert!(wrapped.slot_for(1023).is_err());
        assert!(wrapped.plan_gather(1_033, 4_096).is_ok());
    }

    #[test]
    fn marconi_skip_is_idempotent_at_current_cursor_and_rejects_rewind() {
        let skipped = RingState::new(4096, 256)
            .unwrap()
            .advance_skipped_prefix(512)
            .unwrap();
        assert_eq!(skipped.advance_skipped_prefix(512).unwrap(), skipped);
        assert!(skipped.advance_skipped_prefix(511).is_err());
        assert!(skipped.advance_skipped_prefix(4097).is_err());
    }
}
