// SPDX-License-Identifier: AGPL-3.0-only

//! Checked placement for metadata retained across a fused decode/prefill loop.

use anyhow::{Result, anyhow, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MixedPrefillScratchLayout {
    pub meta_offset: usize,
    pub positions_bytes: usize,
    pub slot_offset: usize,
    pub slots_bytes: usize,
    pub metadata_bytes: usize,
    pub metadata_end: usize,
}

impl MixedPrefillScratchLayout {
    pub fn new(
        prefill_rows: usize,
        padded_decode_rows: usize,
        top_k: usize,
        scratch_bytes: usize,
    ) -> Result<Self> {
        let overflow = || anyhow!("mixed_forward prefill scratch extent overflow");
        // Adapted from Atlas-Inf/atlas PR162 (20d9b1fb): decode runs first,
        // so a short prefill tail must also clear the decode routing rows.
        let routing_bytes = top_k
            .checked_mul(8)
            .and_then(|row_bytes| prefill_rows.max(padded_decode_rows).checked_mul(row_bytes))
            .ok_or_else(overflow)?;
        let meta_offset = routing_bytes.checked_add(7).ok_or_else(overflow)? & !7;
        let positions_bytes = prefill_rows.checked_mul(4).ok_or_else(overflow)?;
        let slot_offset = positions_bytes.checked_add(7).ok_or_else(overflow)? & !7;
        let slots_bytes = prefill_rows.checked_mul(8).ok_or_else(overflow)?;
        let metadata_bytes = slot_offset.checked_add(slots_bytes).ok_or_else(overflow)?;
        let metadata_end = meta_offset
            .checked_add(metadata_bytes)
            .ok_or_else(overflow)?;
        ensure!(
            metadata_end <= scratch_bytes,
            "mixed_forward prefill metadata ends at byte {metadata_end}, beyond the {scratch_bytes}-byte scratch arena"
        );
        Ok(Self {
            meta_offset,
            positions_bytes,
            slot_offset,
            slots_bytes,
            metadata_bytes,
            metadata_end,
        })
    }

    pub fn upload_bytes(self, needs_paged: bool) -> usize {
        if needs_paged {
            self.positions_bytes
        } else {
            self.metadata_bytes
        }
    }
}
