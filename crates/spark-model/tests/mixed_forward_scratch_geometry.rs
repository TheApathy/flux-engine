// SPDX-License-Identifier: AGPL-3.0-only

//! Exercises the production layout helper without a model or GPU execution.

#[path = "../src/model/trait_impl/mixed_scratch.rs"]
mod mixed_scratch;

use mixed_scratch::MixedPrefillScratchLayout;

#[test]
fn one_token_tail_survives_two_row_routing_writes() {
    let position = 4_096u32.to_ne_bytes();
    let slot = 98_304i64.to_ne_bytes();

    // The historical top-k=8 one-row placement was byte 64. Two-row
    // routing writes 128 bytes, destroying both live metadata fields.
    let mut legacy = [0xCC; 144];
    legacy[64..68].copy_from_slice(&position);
    legacy[72..80].copy_from_slice(&slot);
    legacy[..128].fill(0x7F);
    assert_ne!(&legacy[64..68], &position);
    assert_ne!(&legacy[72..80], &slot);

    let layout = MixedPrefillScratchLayout::new(1, 2, 8, 144).unwrap();
    assert_eq!(layout.meta_offset, 128);
    assert_eq!(layout.positions_bytes, 4);
    assert_eq!(layout.slot_offset, 8);
    assert_eq!(layout.slots_bytes, 8);
    assert_eq!(layout.metadata_bytes, 16);
    assert_eq!(layout.metadata_end, 144);
    let mut scratch = vec![0xCC; layout.metadata_end];
    let position_start = layout.meta_offset;
    let slot_start = layout.meta_offset + layout.slot_offset;
    scratch[position_start..position_start + layout.positions_bytes].copy_from_slice(&position);
    scratch[slot_start..slot_start + layout.slots_bytes].copy_from_slice(&slot);
    let metadata = scratch[layout.meta_offset..].to_vec();
    scratch[..128].fill(0x7F);
    assert_eq!(&scratch[layout.meta_offset..], metadata.as_slice());
    assert_eq!(&scratch[position_start..position_start + 4], &position);
    assert_eq!(&scratch[slot_start..slot_start + 8], &slot);
}

#[test]
fn exact_capacity_and_one_byte_short_boundaries() {
    for (prefill, decode, top_k, meta_offset, metadata_end) in [
        (1, 2, 8, 128, 144),
        (21, 32, 8, 2_048, 2_304),
        (1, 128, 8, 8_192, 8_208),
        (4_096, 32, 8, 262_144, 311_296),
        (1, 2, 0, 0, 16),
    ] {
        let layout = MixedPrefillScratchLayout::new(prefill, decode, top_k, metadata_end).unwrap();
        assert_eq!(layout.meta_offset, meta_offset);
        assert_eq!(layout.metadata_end, metadata_end);
        assert_eq!(layout.meta_offset % 8, 0);
        assert_eq!(layout.slot_offset % 8, 0);
        assert!(MixedPrefillScratchLayout::new(prefill, decode, top_k, metadata_end - 1).is_err());
        assert_eq!(layout.upload_bytes(true), layout.positions_bytes);
        assert_eq!(layout.upload_bytes(false), layout.metadata_bytes);
    }
}

#[test]
fn rejects_routing_position_slot_alignment_and_end_overflows() {
    for (prefill, decode, top_k) in [
        (1, 2, usize::MAX),
        (1, usize::MAX / 64 + 1, 8),
        (usize::MAX / 4 + 1, 1, 0),
        (usize::MAX / 4, 1, 0),
        (usize::MAX / 8 + 1, 1, 0),
        (usize::MAX / 8, 1, 0),
        (1, (usize::MAX - 15) / 8, 1),
    ] {
        let error = MixedPrefillScratchLayout::new(prefill, decode, top_k, usize::MAX).unwrap_err();
        assert!(error.to_string().contains("scratch extent overflow"));
    }
}
