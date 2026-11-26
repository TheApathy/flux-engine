// SPDX-License-Identifier: AGPL-3.0-only

//! Coalesced fixed-size EP/TP step control frame (`<= 256 B`).
//!
//! Defines a validated 224-byte (`<= 256 B`) wire schema designed to replace
//! multi-round per-word `ep_broadcast_u32` control chatter (19 separate `u32`
//! broadcasts for an 8-sequence batched verify header, up to 59 broadcasts when
//! paired with per-slot verify commits) with a single inline-WQE broadcast
//! frame. Currently a standalone, unit-tested schema (not yet wired into live
//! EP dispatch).
//!
//! Serialization uses explicit little-endian byte packing with strict structural
//! validation and zero `unsafe` transmutes.

use anyhow::{Result, bail, ensure};

/// Magic tag `"EPCF"` (`0x4550_4346`) identifying a valid [`EpStepControlFrame`].
pub const EP_CONTROL_FRAME_MAGIC: u32 = 0x4550_4346;

/// Wire schema version for [`EpStepControlFrame`].
pub const EP_CONTROL_FRAME_VERSION: u16 = 1;

/// Maximum per-sequence entries carried in a single [`EpStepControlFrame`].
pub const EP_CONTROL_MAX_SEQS: usize = 16;

/// Serialized size of the fixed header in bytes (`32` B).
pub const EP_CONTROL_FRAME_HEADER_BYTES: usize = 32;

/// Serialized size of one [`EpControlSeqEntry`] in bytes (`12` B).
pub const EP_CONTROL_SEQ_ENTRY_BYTES: usize = 12;

/// Exact wire size of one [`EpStepControlFrame`] (`32 + 16 * 12 = 224` bytes, `<= 256` B).
pub const EP_CONTROL_FRAME_BYTES: usize =
    EP_CONTROL_FRAME_HEADER_BYTES + EP_CONTROL_MAX_SEQS * EP_CONTROL_SEQ_ENTRY_BYTES;

/// Sentinel in [`EpControlSeqEntry::accepted`] marking a per-sequence verify abort.
pub const EP_SEQ_ACCEPTED_ABORT: u16 = u16::MAX;

/// Step-level control frame flags (`EpStepControlFrame::flags`).
pub const EP_FRAME_FLAG_HAS_TOKEN_PAYLOAD: u16 = 1 << 0;
pub const EP_FRAME_FLAG_COMMIT_PHASE: u16 = 1 << 1;
pub const EP_FRAME_FLAG_ABORT_PHASE: u16 = 1 << 2;
pub const EP_FRAME_FLAG_FREE_SLOTS: u16 = 1 << 3;
pub const EP_FRAME_FLAG_SHUTDOWN: u16 = 1 << 4;

/// Per-sequence entry flags (low byte of [`EpControlSeqEntry::seq_flags`]).
pub const EP_SEQ_FLAG_QSA_REQUIRED: u16 = 1 << 0;
pub const EP_SEQ_FLAG_FRESH_IF_DIRTY: u16 = 1 << 1;
pub const EP_SEQ_FLAG_ABORT_LANE: u16 = 1 << 2;

/// Per-sequence control descriptor carried inside [`EpStepControlFrame`].
///
/// Packs `(slot, k, accepted)` alongside sequence flags/tail count and `seq_len`
/// with zero internal padding (`12` bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EpControlSeqEntry {
    /// Sequence slot index in the worker's SSM/KV pool.
    pub slot: u16,
    /// Active verification/decode window width (`k` rows; `1` for single-token decode).
    pub k: u16,
    /// Accepted token count for commit steps (`0..=k`), or [`EP_SEQ_ACCEPTED_ABORT`] on abort.
    pub accepted: u16,
    /// Low byte: `EP_SEQ_FLAG_*` bits; high byte: opcode-specific `tail_len` or `last_inter_slot`.
    pub seq_flags: u16,
    /// Logical sequence length on rank 0 prior to this step.
    pub seq_len: u32,
}

impl EpControlSeqEntry {
    /// Construct a per-sequence entry from `(slot, k, accepted)` with default flags and length.
    #[must_use]
    pub const fn new(slot: u16, k: u16, accepted: u16) -> Self {
        Self {
            slot,
            k,
            accepted,
            seq_flags: 0,
            seq_len: 0,
        }
    }

    /// Pack `(qsa_required, fresh_if_dirty, abort_lane, aux_u8)` into `seq_flags`.
    #[must_use]
    pub const fn pack_flags(
        qsa_required: bool,
        fresh_if_dirty: bool,
        abort_lane: bool,
        aux_u8: u8,
    ) -> u16 {
        let mut f = (aux_u8 as u16) << 8;
        if qsa_required {
            f |= EP_SEQ_FLAG_QSA_REQUIRED;
        }
        if fresh_if_dirty {
            f |= EP_SEQ_FLAG_FRESH_IF_DIRTY;
        }
        if abort_lane {
            f |= EP_SEQ_FLAG_ABORT_LANE;
        }
        f
    }

    /// Extract the auxiliary high byte (`tail_len` or `last_inter_slot`) from `seq_flags`.
    #[must_use]
    pub const fn aux_u8(&self) -> u8 {
        (self.seq_flags >> 8) as u8
    }

    /// Returns `true` when this sequence requires QSA routing.
    #[must_use]
    pub const fn qsa_required(&self) -> bool {
        (self.seq_flags & EP_SEQ_FLAG_QSA_REQUIRED) != 0
    }

    /// Returns `true` when this sequence slot should be reset if dirty.
    #[must_use]
    pub const fn fresh_if_dirty(&self) -> bool {
        (self.seq_flags & EP_SEQ_FLAG_FRESH_IF_DIRTY) != 0
    }

    /// Returns `true` when this sequence lane is marked for verify abort.
    #[must_use]
    pub const fn is_abort(&self) -> bool {
        self.accepted == EP_SEQ_ACCEPTED_ABORT || (self.seq_flags & EP_SEQ_FLAG_ABORT_LANE) != 0
    }
}

/// Fixed-size (`224 B <= 256 B`) coalesced EP/TP step control frame broadcast once per step.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpStepControlFrame {
    /// Magic header tag (must equal [`EP_CONTROL_FRAME_MAGIC`]).
    pub magic: u32,
    /// Wire format version (must equal [`EP_CONTROL_FRAME_VERSION`]).
    pub version: u16,
    /// Bitmask of `EP_FRAME_FLAG_*` step control flags.
    pub flags: u16,
    /// Command opcode (e.g. `EP_CMD_VERIFY_QWEN4_BATCHED`, `EP_CMD_COMMIT_VERIFY_SLOT`,
    /// prefill `0xFFFFFFF0`, free `0xFFFFFFF1`, shutdown `0xFFFFFFFF`, or decode token).
    pub opcode: u32,
    /// Number of valid entries in `seqs[..batch_size]` (`0..=EP_CONTROL_MAX_SEQS`).
    pub batch_size: u16,
    /// Reserved header field (must be `0` in version 1).
    pub reserved: u16,
    /// Monotonic step sequence number for lockstep rank diagnostics.
    pub step_id: u32,
    /// Opcode-specific argument 0 (e.g. `chunk_len` or shared verify `k`).
    pub aux_arg0: u32,
    /// Opcode-specific argument 1 (e.g. `chunk_start` or `total_tail_tokens`).
    pub aux_arg1: u32,
    /// Opcode-specific argument 2 (e.g. `full_len` or token payload count).
    pub aux_arg2: u32,
    /// Fixed array of per-sequence `(slot, k, accepted)` descriptors.
    pub seqs: [EpControlSeqEntry; EP_CONTROL_MAX_SEQS],
}

const _: () = assert!(core::mem::size_of::<EpControlSeqEntry>() == EP_CONTROL_SEQ_ENTRY_BYTES);
const _: () = assert!(core::mem::size_of::<EpStepControlFrame>() == EP_CONTROL_FRAME_BYTES);
const _: () = assert!(core::mem::size_of::<EpStepControlFrame>() <= 256);

impl EpStepControlFrame {
    /// Construct an empty control frame header with zero active sequence entries.
    #[must_use]
    pub const fn new(opcode: u32, flags: u16, step_id: u32) -> Self {
        Self {
            magic: EP_CONTROL_FRAME_MAGIC,
            version: EP_CONTROL_FRAME_VERSION,
            flags,
            opcode,
            batch_size: 0,
            reserved: 0,
            step_id,
            aux_arg0: 0,
            aux_arg1: 0,
            aux_arg2: 0,
            seqs: [EpControlSeqEntry::new(0, 0, 0); EP_CONTROL_MAX_SEQS],
        }
    }

    /// Append one per-sequence descriptor to this frame.
    ///
    /// # Errors
    /// Returns an error if `batch_size` has already reached [`EP_CONTROL_MAX_SEQS`] or if
    /// `entry` has an invalid `accepted > k` count.
    pub fn push_seq(&mut self, entry: EpControlSeqEntry) -> Result<()> {
        let idx = self.batch_size as usize;
        ensure!(
            idx < EP_CONTROL_MAX_SEQS,
            "EpStepControlFrame batch_size exceeds maximum ({EP_CONTROL_MAX_SEQS})"
        );
        validate_seq_entry(idx, &entry)?;
        self.seqs[idx] = entry;
        self.batch_size += 1;
        Ok(())
    }

    /// Slice of active per-sequence entries (`0..batch_size`).
    #[must_use]
    pub fn active_seqs(&self) -> &[EpControlSeqEntry] {
        let count = (self.batch_size as usize).min(EP_CONTROL_MAX_SEQS);
        &self.seqs[..count]
    }

    /// Validate all header fields, active sequence bounds, and zeroed inactive tail slots.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.magic == EP_CONTROL_FRAME_MAGIC,
            "invalid EpStepControlFrame magic: expected {EP_CONTROL_FRAME_MAGIC:#010x}, got {:#010x}",
            self.magic
        );
        ensure!(
            self.version == EP_CONTROL_FRAME_VERSION,
            "unsupported EpStepControlFrame version: expected {EP_CONTROL_FRAME_VERSION}, got {}",
            self.version
        );
        ensure!(
            self.reserved == 0,
            "non-zero EpStepControlFrame reserved header field: {:#06x}",
            self.reserved
        );
        let count = self.batch_size as usize;
        ensure!(
            count <= EP_CONTROL_MAX_SEQS,
            "EpStepControlFrame batch_size {count} exceeds max {EP_CONTROL_MAX_SEQS}"
        );
        for (i, entry) in self.seqs[..count].iter().enumerate() {
            validate_seq_entry(i, entry)?;
        }
        let zero_entry = EpControlSeqEntry::default();
        for (i, entry) in self.seqs[count..].iter().enumerate() {
            if *entry != zero_entry {
                bail!(
                    "non-zero inactive EpStepControlFrame entry at index {}",
                    count + i
                );
            }
        }
        Ok(())
    }

    /// Serialize this frame into its canonical little-endian 224-byte wire buffer (zero `unsafe`).
    ///
    /// # Errors
    /// Returns an error if [`Self::validate`] fails.
    pub fn to_bytes(self) -> Result<[u8; EP_CONTROL_FRAME_BYTES]> {
        self.validate()?;
        let mut out = [0u8; EP_CONTROL_FRAME_BYTES];
        write_u32_le(&mut out, 0, self.magic);
        write_u16_le(&mut out, 4, self.version);
        write_u16_le(&mut out, 6, self.flags);
        write_u32_le(&mut out, 8, self.opcode);
        write_u16_le(&mut out, 12, self.batch_size);
        write_u16_le(&mut out, 14, self.reserved);
        write_u32_le(&mut out, 16, self.step_id);
        write_u32_le(&mut out, 20, self.aux_arg0);
        write_u32_le(&mut out, 24, self.aux_arg1);
        write_u32_le(&mut out, 28, self.aux_arg2);

        for (i, entry) in self.seqs.iter().enumerate() {
            let base = EP_CONTROL_FRAME_HEADER_BYTES + i * EP_CONTROL_SEQ_ENTRY_BYTES;
            write_u16_le(&mut out, base, entry.slot);
            write_u16_le(&mut out, base + 2, entry.k);
            write_u16_le(&mut out, base + 4, entry.accepted);
            write_u16_le(&mut out, base + 6, entry.seq_flags);
            write_u32_le(&mut out, base + 8, entry.seq_len);
        }
        Ok(out)
    }

    /// Deserialize and validate an [`EpStepControlFrame`] from little-endian wire bytes (zero `unsafe`).
    ///
    /// # Errors
    /// Returns an error if `bytes.len() != EP_CONTROL_FRAME_BYTES` or any structural
    /// invariant checked by [`Self::validate`] fails.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() == EP_CONTROL_FRAME_BYTES,
            "EpStepControlFrame length mismatch: expected {EP_CONTROL_FRAME_BYTES} bytes, got {}",
            bytes.len()
        );
        let mut seqs = [EpControlSeqEntry::default(); EP_CONTROL_MAX_SEQS];
        for (i, dst) in seqs.iter_mut().enumerate() {
            let base = EP_CONTROL_FRAME_HEADER_BYTES + i * EP_CONTROL_SEQ_ENTRY_BYTES;
            *dst = EpControlSeqEntry {
                slot: read_u16_le(bytes, base)?,
                k: read_u16_le(bytes, base + 2)?,
                accepted: read_u16_le(bytes, base + 4)?,
                seq_flags: read_u16_le(bytes, base + 6)?,
                seq_len: read_u32_le(bytes, base + 8)?,
            };
        }
        let frame = Self {
            magic: read_u32_le(bytes, 0)?,
            version: read_u16_le(bytes, 4)?,
            flags: read_u16_le(bytes, 6)?,
            opcode: read_u32_le(bytes, 8)?,
            batch_size: read_u16_le(bytes, 12)?,
            reserved: read_u16_le(bytes, 14)?,
            step_id: read_u32_le(bytes, 16)?,
            aux_arg0: read_u32_le(bytes, 20)?,
            aux_arg1: read_u32_le(bytes, 24)?,
            aux_arg2: read_u32_le(bytes, 28)?,
            seqs,
        };
        frame.validate()?;
        Ok(frame)
    }
}

/// Legacy count of separate `ep_broadcast_u32` calls for a batched verify header alone
/// (`cmd + n_seqs + k + 2 * batch_size`, e.g. `19` for `batch_size = 8`).
#[must_use]
pub const fn legacy_u32_broadcasts_for_verify_header(batch_size: usize) -> usize {
    3 + 2 * batch_size
}

/// Legacy count of separate `ep_broadcast_u32` calls for a batched verify header plus
/// per-slot commit frames (`3 + 2 * B + 5 * B = 3 + 7 * B`, e.g. `59` for `batch_size = 8`).
#[must_use]
pub const fn legacy_u32_broadcasts_for_verify_and_commit(batch_size: usize) -> usize {
    legacy_u32_broadcasts_for_verify_header(batch_size) + 5 * batch_size
}

fn validate_seq_entry(idx: usize, entry: &EpControlSeqEntry) -> Result<()> {
    if entry.accepted != EP_SEQ_ACCEPTED_ABORT {
        ensure!(
            entry.accepted <= entry.k,
            "EpStepControlFrame seq[{idx}] accepted ({}) exceeds k ({})",
            entry.accepted,
            entry.k
        );
    }
    Ok(())
}

#[inline]
fn write_u16_le(buf: &mut [u8], off: usize, val: u16) {
    buf[off..off + 2].copy_from_slice(&val.to_le_bytes());
}

#[inline]
fn write_u32_le(buf: &mut [u8], off: usize, val: u32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}

#[inline]
fn read_u16_le(buf: &[u8], off: usize) -> Result<u16> {
    let slice = buf
        .get(off..off + 2)
        .ok_or_else(|| anyhow::anyhow!("truncated u16 at offset {off}"))?;
    let arr: [u8; 2] = slice
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid u16 slice length at offset {off}"))?;
    Ok(u16::from_le_bytes(arr))
}

#[inline]
fn read_u32_le(buf: &[u8], off: usize) -> Result<u32> {
    let slice = buf
        .get(off..off + 4)
        .ok_or_else(|| anyhow::anyhow!("truncated u32 at offset {off}"))?;
    let arr: [u8; 4] = slice
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid u32 slice length at offset {off}"))?;
    Ok(u32::from_le_bytes(arr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{EP_CMD_COMMIT_VERIFY_SLOT, EP_CMD_VERIFY_QWEN4_BATCHED};
    use parking_lot::Mutex;
    use spark_comm::CommBackend;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_control_frame_size_within_256b_budget() {
        assert_eq!(core::mem::size_of::<EpControlSeqEntry>(), 12);
        assert_eq!(core::mem::size_of::<EpStepControlFrame>(), 224);
        assert!(core::mem::size_of::<EpStepControlFrame>() <= 256);
    }

    #[test]
    fn test_control_frame_le_roundtrip_batched_verify_and_commit() {
        let mut frame = EpStepControlFrame::new(
            EP_CMD_VERIFY_QWEN4_BATCHED,
            EP_FRAME_FLAG_HAS_TOKEN_PAYLOAD | EP_FRAME_FLAG_COMMIT_PHASE,
            42,
        );
        frame.aux_arg0 = 5; // shared k = 5
        frame.aux_arg1 = 24; // total_tail_tokens
        frame.aux_arg2 = 40; // flat_window_tokens (8 * 5)

        for i in 0..8u16 {
            let accepted = if i == 3 {
                EP_SEQ_ACCEPTED_ABORT
            } else {
                (i % 5) + 1
            };
            let entry = EpControlSeqEntry {
                slot: i * 2 + 1,
                k: 5,
                accepted,
                seq_flags: EpControlSeqEntry::pack_flags(
                    i.is_multiple_of(2),
                    i == 0,
                    i == 3,
                    (i + 3) as u8,
                ),
                seq_len: 1024 + u32::from(i) * 17,
            };
            frame.push_seq(entry).expect("push within max 16 seqs");
        }
        assert_eq!(frame.active_seqs().len(), 8);

        let wire = frame.to_bytes().expect("valid frame serializes");
        assert_eq!(wire.len(), EP_CONTROL_FRAME_BYTES);
        let decoded = EpStepControlFrame::from_bytes(&wire).expect("valid wire bytes deserialize");
        assert_eq!(decoded, frame);
        assert!(decoded.active_seqs()[0].qsa_required());
        assert!(decoded.active_seqs()[0].fresh_if_dirty());
        assert_eq!(decoded.active_seqs()[0].aux_u8(), 3);
        assert!(decoded.active_seqs()[3].is_abort());
        assert!(!decoded.active_seqs()[1].is_abort());
    }

    #[test]
    fn test_control_frame_validation_rejects_corrupt_wire_inputs() {
        let mut frame = EpStepControlFrame::new(EP_CMD_COMMIT_VERIFY_SLOT, 0, 7);
        frame
            .push_seq(EpControlSeqEntry::new(2, 4, 3))
            .expect("valid entry");
        let valid_wire = frame.to_bytes().expect("valid wire");

        // Wrong length
        assert!(EpStepControlFrame::from_bytes(&valid_wire[..223]).is_err());

        // Bad magic
        let mut bad_magic = valid_wire;
        bad_magic[0] ^= 0xFF;
        assert!(EpStepControlFrame::from_bytes(&bad_magic).is_err());

        // Unsupported version
        let mut bad_ver = valid_wire;
        bad_ver[4..6].copy_from_slice(&99u16.to_le_bytes());
        assert!(EpStepControlFrame::from_bytes(&bad_ver).is_err());

        // Non-zero reserved bytes
        let mut bad_res = valid_wire;
        bad_res[14..16].copy_from_slice(&1u16.to_le_bytes());
        assert!(EpStepControlFrame::from_bytes(&bad_res).is_err());

        // batch_size > EP_CONTROL_MAX_SEQS
        let mut bad_batch = valid_wire;
        bad_batch[12..14].copy_from_slice(&((EP_CONTROL_MAX_SEQS as u16) + 1).to_le_bytes());
        assert!(EpStepControlFrame::from_bytes(&bad_batch).is_err());

        // accepted > k (not abort sentinel)
        let mut bad_acc = valid_wire;
        let entry0_acc_off = EP_CONTROL_FRAME_HEADER_BYTES + 4;
        bad_acc[entry0_acc_off..entry0_acc_off + 2].copy_from_slice(&5u16.to_le_bytes());
        assert!(EpStepControlFrame::from_bytes(&bad_acc).is_err());

        // Non-zero inactive tail entry
        let mut bad_tail = valid_wire;
        let inactive_off = EP_CONTROL_FRAME_HEADER_BYTES + EP_CONTROL_SEQ_ENTRY_BYTES;
        bad_tail[inactive_off] = 1;
        assert!(EpStepControlFrame::from_bytes(&bad_tail).is_err());

        // Pushing past capacity fails cleanly.
        let mut full = EpStepControlFrame::new(EP_CMD_VERIFY_QWEN4_BATCHED, 0, 1);
        assert!(
            full.push_seq(EpControlSeqEntry::new(0, 0, 99)).is_err(),
            "k == 0 must still reject accepted > 0 unless EP_SEQ_ACCEPTED_ABORT"
        );
        for slot in 0..EP_CONTROL_MAX_SEQS as u16 {
            full.push_seq(EpControlSeqEntry::new(slot, 4, 2))
                .expect("slot fits");
        }
        assert!(full.push_seq(EpControlSeqEntry::new(99, 4, 2)).is_err());
    }

    #[test]
    fn test_one_broadcast_coalescing_replaces_19_to_59_u32_broadcasts() {
        // Verify exact legacy broadcast counts for C=8 batched verify (19) and verify+commit (59).
        assert_eq!(legacy_u32_broadcasts_for_verify_header(8), 19);
        assert_eq!(legacy_u32_broadcasts_for_verify_and_commit(8), 59);
        assert_eq!(legacy_u32_broadcasts_for_verify_header(16), 35);
        assert_eq!(legacy_u32_broadcasts_for_verify_and_commit(16), 115);

        struct SingleBroadcastMockComm {
            broadcast_calls: AtomicUsize,
            tx_bufs: Mutex<std::collections::HashMap<u64, Vec<u8>>>,
            rx_wire_buf: Mutex<Vec<u8>>,
        }
        impl CommBackend for SingleBroadcastMockComm {
            fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()> {
                assert_eq!(root, 0);
                assert_eq!(bytes, EP_CONTROL_FRAME_BYTES);
                self.broadcast_calls.fetch_add(1, Ordering::SeqCst);
                let tx = self.tx_bufs.lock();
                let payload = tx
                    .get(&ptr)
                    .ok_or_else(|| anyhow::anyhow!("unknown broadcast ptr {ptr:#x}"))?;
                ensure!(
                    payload.len() == bytes,
                    "broadcast byte count mismatch: {} != {bytes}",
                    payload.len()
                );
                let mut rx = self.rx_wire_buf.lock();
                rx.clear();
                rx.extend_from_slice(payload);
                Ok(())
            }
            fn barrier(&self) -> Result<()> {
                Ok(())
            }
            fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
                Ok(())
            }
            fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
                Ok(())
            }
            fn rank(&self) -> usize {
                0
            }
            fn world_size(&self) -> usize {
                4
            }
        }

        // Pack all 8 sequences' verify + commit metadata into ONE control frame.
        let mut frame =
            EpStepControlFrame::new(EP_CMD_VERIFY_QWEN4_BATCHED, EP_FRAME_FLAG_COMMIT_PHASE, 100);
        frame.aux_arg0 = 5;
        for slot in 0..8u16 {
            frame
                .push_seq(EpControlSeqEntry {
                    slot,
                    k: 5,
                    accepted: (slot % 5) + 1,
                    seq_flags: EpControlSeqEntry::pack_flags(true, false, false, 4),
                    seq_len: 512 + u32::from(slot),
                })
                .expect("8 seqs fit in one frame");
        }
        let wire = frame.to_bytes().expect("serialize coalesced frame");

        const TX_PTR: u64 = 0x1000;
        let comm = SingleBroadcastMockComm {
            broadcast_calls: AtomicUsize::new(0),
            tx_bufs: Mutex::new(std::collections::HashMap::from([(TX_PTR, wire.to_vec())])),
            rx_wire_buf: Mutex::new(Vec::new()),
        };
        assert!(
            EpStepControlFrame::from_bytes(&comm.rx_wire_buf.lock()).is_err(),
            "receiver buffer must start empty before broadcast"
        );
        comm.broadcast(TX_PTR, wire.len(), 0)
            .expect("single coalesced broadcast succeeds");
        assert_eq!(
            comm.broadcast_calls.load(Ordering::SeqCst),
            1,
            "1 coalesced 224B broadcast replaces 19..59 separate u32 broadcasts"
        );

        let worker_frame = EpStepControlFrame::from_bytes(&comm.rx_wire_buf.lock())
            .expect("worker decodes coalesced frame");
        assert_eq!(worker_frame, frame);
        assert_eq!(worker_frame.active_seqs().len(), 8);
    }
}
