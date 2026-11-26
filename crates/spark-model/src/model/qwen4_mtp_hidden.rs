// SPDX-License-Identifier: AGPL-3.0-only

//! Checked, sequence-owned native-MTP target-hidden handoff.
//!
//! This owns only retained hidden rows, not target/proposer execution scratch
//! or recurrent/QSA transactions. It does not admit speculative native batching.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ByteRange {
    pub offset: usize,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrameOwner {
    session: u64,
    position: usize,
    selected_row: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HiddenRead {
    pub prefix: ByteRange,
    pub selected: ByteRange,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct HiddenWrite {
    slot: usize,
    epoch: u64,
    owner: FrameOwner,
    pub retained: HiddenRead,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct HiddenMove {
    old_slot: usize,
    new_slot: usize,
    old_epoch: u64,
    new_epoch: u64,
    owner: Option<FrameOwner>,
    pub copy: Option<(ByteRange, ByteRange)>,
}

pub(crate) struct NativeMtpHiddenFrames {
    rows: usize,
    row_bytes: usize,
    slot_bytes: usize,
    allocation_bytes: usize,
    owners: Vec<Option<FrameOwner>>,
    epochs: Vec<u64>,
}

type Checked<T> = Result<T, &'static str>;

impl NativeMtpHiddenFrames {
    pub fn new(slots: usize, rows: usize, row_bytes: usize) -> Checked<Self> {
        if slots == 0 || slots > 128 || rows == 0 || row_bytes == 0 || !row_bytes.is_multiple_of(2)
        {
            return Err("native MTP hidden storage requires nonzero BF16 geometry");
        }
        // This is retained-storage capacity, not a kernel's active row count.
        // The production SSM authority includes a spare intermediate row.
        let slot_bytes = rows
            .checked_mul(row_bytes)
            .ok_or("native MTP hidden slot extent overflow")?;
        let allocation_bytes = slots
            .checked_mul(slot_bytes)
            .ok_or("native MTP hidden allocation extent overflow")?;
        if allocation_bytes > isize::MAX as usize {
            return Err("native MTP hidden allocation exceeds pointer extent");
        }
        Ok(Self {
            rows,
            row_bytes,
            slot_bytes,
            allocation_bytes,
            owners: vec![None; slots],
            epochs: vec![0; slots],
        })
    }

    pub fn allocation_bytes(&self) -> usize {
        self.allocation_bytes
    }

    pub(crate) fn slot_range(&self, slot: usize) -> Checked<ByteRange> {
        if slot >= self.owners.len() {
            return Err("native MTP hidden slot is outside retained storage");
        }
        Ok(ByteRange {
            offset: slot * self.slot_bytes,
            bytes: self.slot_bytes,
        })
    }

    fn read_range(&self, slot: usize, selected_row: usize) -> Checked<HiddenRead> {
        if slot >= self.owners.len() || selected_row >= self.rows {
            return Err("native MTP hidden slot or row is outside retained storage");
        }
        // Constructor-checked total geometry bounds both products and sums.
        let offset = slot * self.slot_bytes;
        Ok(HiddenRead {
            prefix: ByteRange {
                offset,
                bytes: (selected_row + 1) * self.row_bytes,
            },
            selected: ByteRange {
                offset: offset + selected_row * self.row_bytes,
                bytes: self.row_bytes,
            },
        })
    }

    fn next_epoch(&self, slot: usize) -> Checked<u64> {
        self.epochs
            .get(slot)
            .ok_or("native MTP hidden slot is outside retained storage")?
            .checked_add(1)
            .ok_or("native MTP hidden slot epoch exhausted")
    }

    /// Invalidates the old frame before device work. Only publish after every
    /// copy was enqueued successfully on the same stream used by its consumer.
    pub fn begin_write(
        &mut self,
        slot: usize,
        session: u64,
        position: usize,
        selected_row: usize,
        source_bytes: usize,
    ) -> Checked<HiddenWrite> {
        let retained = self.read_range(slot, selected_row)?;
        if retained.prefix.bytes > source_bytes {
            return Err("native MTP hidden prefix exceeds the target hidden arena");
        }
        let epoch = self.next_epoch(slot)?;
        self.owners[slot] = None;
        self.epochs[slot] = epoch;
        Ok(HiddenWrite {
            slot,
            epoch,
            owner: FrameOwner {
                session,
                position,
                selected_row,
            },
            retained,
        })
    }

    pub fn publish(&mut self, write: HiddenWrite) -> Checked<()> {
        if self.epochs.get(write.slot) != Some(&write.epoch) || self.owners[write.slot].is_some() {
            return Err("native MTP hidden write no longer owns its slot");
        }
        self.owners[write.slot] = Some(write.owner);
        Ok(())
    }

    /// A handoff is consumed once. Stale cursor/session bindings cannot read
    /// a previous request's retained row even after physical slot reuse.
    pub fn take(&mut self, slot: usize, session: u64, position: usize) -> Checked<HiddenRead> {
        let owner = self
            .owners
            .get(slot)
            .copied()
            .flatten()
            .ok_or("native MTP target hidden frame was not saved for this sequence")?;
        if owner.session != session || owner.position != position {
            return Err("native MTP target hidden frame owner or cursor mismatch");
        }
        let retained = self.read_range(slot, owner.selected_row)?;
        self.owners[slot] = None;
        Ok(retained)
    }

    /// `SeqStateBlob` export: the retained frame of `slot` without consuming
    /// it, as `(position, selected_row, prefix)`. `None` when the slot holds
    /// no frame; an error when the frame belongs to another session.
    pub(crate) fn retained_for_export(
        &self,
        slot: usize,
        session: u64,
    ) -> Checked<Option<(usize, usize, ByteRange)>> {
        let Some(owner) = self
            .owners
            .get(slot)
            .copied()
            .ok_or("native MTP hidden slot is outside retained storage")?
        else {
            return Ok(None);
        };
        if owner.session != session {
            return Err("native MTP hidden frame belongs to another session");
        }
        let retained = self.read_range(slot, owner.selected_row)?;
        Ok(Some((owner.position, owner.selected_row, retained.prefix)))
    }

    pub fn retire(&mut self, slot: usize) -> Checked<()> {
        let epoch = self.next_epoch(slot)?;
        self.owners[slot] = None;
        self.epochs[slot] = epoch;
        Ok(())
    }

    /// The caller holds the pool lock through copy and synchronization. The
    /// destination binding is invalid before the first asynchronous write.
    pub fn begin_move(
        &mut self,
        old_slot: usize,
        new_slot: usize,
        session: u64,
    ) -> Checked<HiddenMove> {
        if old_slot == new_slot || old_slot >= self.owners.len() {
            return Err("native MTP hidden move requires distinct valid slots");
        }
        let old_epoch = self.next_epoch(old_slot)?;
        let new_epoch = self.next_epoch(new_slot)?;
        let owner = self.owners[old_slot];
        if owner.is_some_and(|owner| owner.session != session) {
            return Err("native MTP hidden move source owner mismatch");
        }
        let copy = owner
            .map(|owner| -> Checked<_> {
                Ok((
                    self.read_range(old_slot, owner.selected_row)?.prefix,
                    self.read_range(new_slot, owner.selected_row)?.prefix,
                ))
            })
            .transpose()?;
        self.owners[new_slot] = None;
        self.epochs[new_slot] = new_epoch;
        self.epochs[old_slot] = old_epoch;
        Ok(HiddenMove {
            old_slot,
            new_slot,
            old_epoch,
            new_epoch,
            owner,
            copy,
        })
    }

    pub fn finish_move(&mut self, movement: HiddenMove) -> Checked<()> {
        if self.epochs.get(movement.old_slot) != Some(&movement.old_epoch)
            || self.epochs.get(movement.new_slot) != Some(&movement.new_epoch)
            || self.owners[movement.old_slot] != movement.owner
            || self.owners[movement.new_slot].is_some()
        {
            return Err("native MTP hidden move no longer owns its slots");
        }
        self.owners[movement.old_slot] = None;
        self.owners[movement.new_slot] = movement.owner;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn save(
        pool: &mut NativeMtpHiddenFrames,
        slot: usize,
        session: u64,
        position: usize,
        row: usize,
    ) -> HiddenWrite {
        let write = pool
            .begin_write(slot, session, position, row, 1024)
            .unwrap();
        pool.publish(write).unwrap();
        write
    }

    #[test]
    fn seqstate_export_peeks_and_import_republishes_into_another_slot() {
        let mut pool = NativeMtpHiddenFrames::new(4, 5, 20).unwrap();
        assert_eq!(pool.retained_for_export(1, 9).unwrap(), None);
        save(&mut pool, 1, 9, 300, 3);
        assert!(pool.retained_for_export(1, 8).is_err(), "foreign session");
        let (position, row, prefix) = pool.retained_for_export(1, 9).unwrap().unwrap();
        assert_eq!((position, row), (300, 3));
        assert_eq!(
            prefix,
            ByteRange {
                offset: 100,
                bytes: 80
            }
        );
        // Peeking does not consume: the in-place owner can still take it.
        assert_eq!(pool.retained_for_export(1, 9).unwrap().unwrap().0, 300);
        // Import publishes the same binding into the destination slot.
        let write = pool.begin_write(2, 9, position, row, prefix.bytes).unwrap();
        assert_eq!(write.retained.prefix.bytes, prefix.bytes);
        pool.publish(write).unwrap();
        assert_eq!(pool.take(2, 9, 300).unwrap().selected.offset, 200 + 60);
        assert!(pool.retained_for_export(9, 9).is_err(), "slot out of range");
    }

    #[test]
    fn c_by_k_hidden_rows_are_disjoint_at_every_width() {
        for slots in [1, 2, 4, 8, 16, 32, 128] {
            let mut pool = NativeMtpHiddenFrames::new(slots, 5, 20).unwrap();
            assert_eq!(pool.allocation_bytes(), slots * 5 * 20);
            for slot in 0..slots {
                let row = slot % 5;
                let write = save(&mut pool, slot, slot as u64, 700 + slot, row);
                assert_eq!(write.retained.prefix.offset, slot * 100);
                assert_eq!(write.retained.prefix.bytes, (row + 1) * 20);
                assert_eq!(write.retained.selected.offset, slot * 100 + row * 20);
                assert!(
                    write.retained.prefix.offset + write.retained.prefix.bytes <= (slot + 1) * 100
                );
            }
            for slot in (0..slots).rev() {
                let read = pool.take(slot, slot as u64, 700 + slot).unwrap();
                assert_eq!(read.selected.offset, slot * 100 + slot % 5 * 20);
                assert!(pool.take(slot, slot as u64, 700 + slot).is_err());
            }
        }
    }

    #[test]
    fn interleaved_frames_keep_each_sessions_actual_prefix_bytes() {
        for slots in [2, 4, 8, 16, 32] {
            let mut pool = NativeMtpHiddenFrames::new(slots, 5, 20).unwrap();
            let mut device = vec![0xFF; pool.allocation_bytes()];
            let mut target_scratch = [0; 100];
            for slot in 0..slots {
                for row in 0..5 {
                    target_scratch[row * 20..(row + 1) * 20].fill((slot * 5 + row) as u8);
                }
                let selected = slot % 5;
                let write = pool
                    .begin_write(slot, slot as u64, 90 + slot, selected, target_scratch.len())
                    .unwrap();
                let range = write.retained.prefix;
                device[range.offset..range.offset + range.bytes]
                    .copy_from_slice(&target_scratch[..range.bytes]);
                pool.publish(write).unwrap();
                // The next target/proposer is free to overwrite every shared
                // scratch byte without changing any retained request prefix.
                target_scratch.fill(0xEE);
            }
            for slot in (0..slots).rev() {
                let retained = pool.take(slot, slot as u64, 90 + slot).unwrap();
                for row in 0..=slot % 5 {
                    let start = retained.prefix.offset + row * 20;
                    assert_eq!(&device[start..start + 20], &[(slot * 5 + row) as u8; 20]);
                }
                assert_eq!(
                    &device[retained.selected.offset..retained.selected.offset + 20],
                    &[(slot * 5 + slot % 5) as u8; 20]
                );
            }
        }
    }

    #[test]
    fn spare_storage_capacity_accepts_row_31_without_touching_row_32() {
        let mut pool = NativeMtpHiddenFrames::new(2, 33, 20).unwrap();
        let mut retained = vec![0xFF; pool.allocation_bytes()];
        let mut target = vec![0u8; 32 * 20];
        for slot in 0..2 {
            for row in 0..32 {
                target[row * 20..(row + 1) * 20].fill((slot * 64 + row) as u8);
            }
            let write = pool
                .begin_write(slot, slot as u64, 100 + slot, 31, target.len())
                .unwrap();
            let prefix = write.retained.prefix;
            assert_eq!(prefix.bytes, target.len());
            retained[prefix.offset..prefix.offset + prefix.bytes].copy_from_slice(&target);
            pool.publish(write).unwrap();
            target.fill(0xEE);
        }
        for slot in (0..2).rev() {
            let read = pool.take(slot, slot as u64, 100 + slot).unwrap();
            assert_eq!(read.selected.offset, (slot * 33 + 31) * 20);
            assert_eq!(
                &retained[read.selected.offset..read.selected.offset + 20],
                &[(slot * 64 + 31) as u8; 20]
            );
            let spare = (slot * 33 + 32) * 20;
            assert_eq!(&retained[spare..spare + 20], &[0xFF; 20]);
        }
        assert!(pool.begin_write(0, 0, 100, 32, target.len()).is_err());
    }

    #[test]
    fn malformed_geometry_and_source_extents_fail_without_mutation() {
        for (slots, rows, bytes) in [
            (0, 5, 20),
            (2, 0, 20),
            (2, 5, 0),
            (2, 5, 3),
            (usize::MAX, 5, 20),
            (1, usize::MAX, 20),
            (1, 1, isize::MAX as usize + 1),
        ] {
            assert!(NativeMtpHiddenFrames::new(slots, rows, bytes).is_err());
        }
        let mut pool = NativeMtpHiddenFrames::new(2, 5, 20).unwrap();
        save(&mut pool, 0, 1, 7, 4);
        for (slot, row, source) in [
            (2, 0, 100),
            (0, 5, 100),
            (0, usize::MAX, usize::MAX),
            (0, 4, 99),
        ] {
            assert!(pool.begin_write(slot, 2, 8, row, source).is_err());
        }
        assert!(pool.take(0, 1, 7).is_ok());
    }

    #[test]
    fn failed_or_superseded_writes_cannot_resurrect_a_frame() {
        let mut pool = NativeMtpHiddenFrames::new(2, 5, 20).unwrap();
        save(&mut pool, 0, 1, 7, 4);
        let failed = pool.begin_write(0, 1, 8, 2, 100).unwrap();
        assert!(pool.take(0, 1, 7).is_err());
        assert!(pool.take(0, 1, 8).is_err());
        save(&mut pool, 0, 2, 9, 0);
        assert!(pool.publish(failed).is_err());
        assert!(pool.take(0, 1, 9).is_err());
        assert!(pool.take(0, 2, 8).is_err());
        assert!(pool.take(0, 2, 9).is_ok());
    }

    #[test]
    fn retirement_invalidates_even_same_session_same_cursor_reuse() {
        let mut pool = NativeMtpHiddenFrames::new(2, 5, 20).unwrap();
        let old = save(&mut pool, 0, 1, 7, 4);
        pool.retire(0).unwrap();
        assert!(pool.publish(old).is_err());
        assert!(pool.take(0, 1, 7).is_err());
        save(&mut pool, 0, 1, 7, 1);
        assert_eq!(pool.take(0, 1, 7).unwrap().selected.offset, 20);
    }

    #[test]
    fn compaction_moves_only_the_owned_prefix_after_copy_barrier() {
        let mut pool = NativeMtpHiddenFrames::new(4, 5, 20).unwrap();
        save(&mut pool, 3, 44, 123, 3);
        save(&mut pool, 0, 11, 50, 0);
        let movement = pool.begin_move(3, 0, 44).unwrap();
        assert_eq!(
            movement.copy,
            Some((
                ByteRange {
                    offset: 300,
                    bytes: 80
                },
                ByteRange {
                    offset: 0,
                    bytes: 80
                }
            ))
        );
        assert!(pool.take(0, 11, 50).is_err());
        assert!(pool.take(0, 44, 123).is_err());
        pool.finish_move(movement).unwrap();
        assert!(pool.take(3, 44, 123).is_err());
        assert_eq!(pool.take(0, 44, 123).unwrap().selected.offset, 60);
        assert!(pool.finish_move(movement).is_err());
    }

    #[test]
    fn invalid_or_failed_moves_cannot_publish_destination_data() {
        let mut pool = NativeMtpHiddenFrames::new(3, 5, 20).unwrap();
        save(&mut pool, 2, 44, 123, 4);
        assert!(pool.begin_move(2, 0, 99).is_err());
        assert!(pool.begin_move(2, 3, 44).is_err());
        assert!(pool.begin_move(2, 2, 44).is_err());
        let movement = pool.begin_move(2, 0, 44).unwrap();
        assert!(pool.take(0, 44, 123).is_err());
        pool.retire(2).unwrap();
        assert!(pool.finish_move(movement).is_err());
    }

    #[test]
    fn empty_source_compaction_retires_destination_without_reading_bytes() {
        let mut pool = NativeMtpHiddenFrames::new(2, 5, 20).unwrap();
        save(&mut pool, 0, 99, 100, 1);
        let movement = pool.begin_move(1, 0, 44).unwrap();
        assert!(movement.copy.is_none());
        pool.finish_move(movement).unwrap();
        assert!(pool.take(0, 99, 100).is_err());
    }

    #[test]
    fn serial_k5_c8_handoffs_never_cross_slots_or_sessions() {
        // ATLAS_QWEN4_K5_CMULTI: eight requests, K5 (num_intermediates = 6),
        // each saving its accepted row in turn before the next verifies.
        let mut pool = NativeMtpHiddenFrames::new(8, 6, 20).unwrap();
        for step in 0..3usize {
            for slot in 0..8usize {
                let accepted = (slot + step) % 5;
                save(
                    &mut pool,
                    slot,
                    0x100 + slot as u64,
                    40 * step + slot,
                    accepted,
                );
            }
            for slot in 0..8usize {
                let other = (slot + 1) % 8;
                assert!(
                    pool.take(slot, 0x100 + other as u64, 40 * step + slot)
                        .is_err()
                );
                assert!(
                    pool.take(slot, 0x100 + slot as u64, 40 * step + other)
                        .is_err()
                );
                let read = pool
                    .take(slot, 0x100 + slot as u64, 40 * step + slot)
                    .unwrap();
                assert_eq!(read.selected.offset, (slot * 6 + (slot + step) % 5) * 20);
                assert!(read.prefix.offset + read.prefix.bytes <= (slot + 1) * 120);
            }
        }
    }
}
