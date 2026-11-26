// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed-stride batched-decode metadata derived from the configured serving
//! width. The 32-row floor preserves the legacy 0/128/256/512/768 byte layout
//! exactly; wider batches round to the same qualified rung used by decode so
//! padding can never outgrow the metadata allocation.

/// Legacy decode metadata was provisioned for exactly 32 rows.
pub const DECODE_META_MIN_ROWS: usize = 32;

/// Widest verify/decode metadata rung provisioned across GB10 (`<= 128`) and
/// wide-SM (`> 64`-SM) 256-row cross-sequence verify frames.
pub const DECODE_META_MAX_ROWS: usize = 256;

const PADDED_DECODE_LADDER: [usize; 13] = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256];

/// Return the smallest qualified decode width that can hold `rows`.
/// Values above [`DECODE_META_MAX_ROWS`] deliberately fall through so
/// serve admission or the metadata upload rejects them instead of truncating.
#[inline]
pub fn padded_decode_rows(rows: usize) -> usize {
    PADDED_DECODE_LADDER
        .iter()
        .copied()
        .find(|&rung| rung >= rows)
        .unwrap_or(rows)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeMetaLayout {
    rows: usize,
}

impl DecodeMetaLayout {
    pub fn for_max_batch_size(max_batch_size: usize) -> Self {
        Self {
            rows: padded_decode_rows(max_batch_size.max(DECODE_META_MIN_ROWS)),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn positions_off(&self) -> usize {
        0
    }

    /// Reserved i32-width region.  This keeps the upstream layout and leaves
    /// room for per-request routing metadata without moving the slot array.
    pub fn routing_off(&self) -> usize {
        4 * self.rows
    }

    pub fn slots_off(&self) -> usize {
        8 * self.rows
    }

    pub fn seq_lens_off(&self) -> usize {
        16 * self.rows
    }

    pub fn block_table_off(&self) -> usize {
        24 * self.rows
    }

    pub fn meta_bytes(&self, max_blocks: usize) -> usize {
        self.block_table_off() + self.rows * max_blocks * 4
    }

    /// Fallible arena admission, before any metadata allocation or upload.
    pub fn checked_meta_bytes(&self, max_blocks: usize) -> anyhow::Result<usize> {
        let prefix = super::checked_math::product(&[24, self.rows], "decode metadata prefix")?;
        let table = super::checked_math::product(
            &[self.rows, max_blocks, 4],
            "decode metadata block table",
        )?;
        super::checked_math::sum(&[prefix, table], "decode metadata total")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_layout_is_byte_identical_through_c32() {
        for batch in [1usize, 16, 31, 32] {
            let layout = DecodeMetaLayout::for_max_batch_size(batch);
            assert_eq!(layout.rows(), 32);
            assert_eq!(layout.positions_off(), 0);
            assert_eq!(layout.routing_off(), 128);
            assert_eq!(layout.slots_off(), 256);
            assert_eq!(layout.seq_lens_off(), 512);
            assert_eq!(layout.block_table_off(), 768);
            assert_eq!(layout.meta_bytes(257), 768 + 32 * 257 * 4);
        }
    }

    #[test]
    fn widened_layout_is_non_overlapping_through_c256() {
        for (batch, expected_rows) in [
            (33usize, 48usize),
            (48, 48),
            (64, 64),
            (65, 96),
            (96, 96),
            (97, 128),
            (128, 128),
            (129, 192),
            (192, 192),
            (193, 256),
            (256, 256),
        ] {
            let layout = DecodeMetaLayout::for_max_batch_size(batch);
            let rows = layout.rows();
            assert_eq!(rows, expected_rows, "batch={batch}");
            assert_eq!(layout.routing_off(), layout.positions_off() + 4 * rows);
            assert_eq!(layout.slots_off(), layout.routing_off() + 4 * rows);
            assert_eq!(layout.slots_off() % 8, 0);
            assert_eq!(layout.seq_lens_off(), layout.slots_off() + 8 * rows);
            assert_eq!(layout.block_table_off(), layout.seq_lens_off() + 8 * rows);
            assert_eq!(layout.meta_bytes(257), 24 * rows + rows * 257 * 4);
        }
    }

    #[test]
    fn qualification_ceiling_is_c256() {
        assert_eq!(DECODE_META_MAX_ROWS, 256);
        assert_eq!(padded_decode_rows(257), 257);
    }

    #[test]
    fn checked_metadata_preserves_valid_layout_and_rejects_overflow() {
        let layout = DecodeMetaLayout::for_max_batch_size(128);
        assert_eq!(
            layout.checked_meta_bytes(257).unwrap(),
            layout.meta_bytes(257)
        );
        assert!(layout.checked_meta_bytes(usize::MAX).is_err());
        assert!(
            DecodeMetaLayout::for_max_batch_size(usize::MAX)
                .checked_meta_bytes(1)
                .is_err()
        );
    }
}
