// SPDX-License-Identifier: AGPL-3.0-only

//! Physical metadata admission, independent of rotary coordinates.

pub(super) fn valid_physical_query(position: usize, sequence_length: usize) -> bool {
    sequence_length <= u32::MAX as usize && position.checked_add(1) == Some(sequence_length)
}

/// The current batched staging kernel has only four raw rows per page.
/// Admitting one compression group prevents both inter-group overwrites and
/// overwriting the old tail required by a group that crosses a chunk boundary.
/// Wider batches need a different pooling/staging transaction, not a fallback.
pub(super) fn prefill_index_batch_is_safe(num_tokens: usize, seq_len_start: usize) -> bool {
    num_tokens > 0
        && num_tokens <= 4
        && seq_len_start
            .checked_add(num_tokens)
            .is_some_and(|end| end <= u32::MAX as usize)
        && seq_len_start % 4 + num_tokens <= 4
}

#[cfg(test)]
mod tests {
    const BLOCK_SIZE: usize = 16;
    const ROWS: usize = 5;

    #[derive(Clone, Copy)]
    struct Undo {
        raw_offset: usize,
        raw: i32,
        compressed_offset: Option<usize>,
        compressed: i32,
    }

    fn offsets(slot: usize) -> (usize, Option<usize>) {
        let block = slot / BLOCK_SIZE;
        let in_block = slot % BLOCK_SIZE;
        let ring = in_block % 4;
        let group = in_block / 4;
        (
            block * 4 + ring,
            (ring == 3).then_some(block * (BLOCK_SIZE / 4) + group),
        )
    }

    fn apply_row(raw: &mut [i32], compressed: &mut [i32], slot: usize, value: i32) {
        let (raw_offset, compressed_offset) = offsets(slot);
        raw[raw_offset] = value;
        if let Some(offset) = compressed_offset {
            let block_base = (slot / BLOCK_SIZE) * 4;
            compressed[offset] = raw[block_base..block_base + 4].iter().sum();
        }
    }

    #[test]
    fn reverse_k5_rollback_matches_only_the_accepted_prefix() {
        // Exercise every ring phase plus physical-page crossings. Row four
        // always aliases row zero modulo four; starts 12..15 cross a page.
        for start in 0..=2 * BLOCK_SIZE {
            for accepted in 0..=ROWS {
                let mut raw: Vec<i32> = (0..16).map(|i| 1_000 + i).collect();
                let mut compressed: Vec<i32> = (0..16).map(|i| 2_000 + i).collect();
                let initial_raw = raw.clone();
                let initial_compressed = compressed.clone();
                let mut undo = Vec::with_capacity(ROWS);

                for row in 0..ROWS {
                    let slot = start + row;
                    let (raw_offset, compressed_offset) = offsets(slot);
                    undo.push(Undo {
                        raw_offset,
                        raw: raw[raw_offset],
                        compressed_offset,
                        compressed: compressed_offset.map_or(0, |offset| compressed[offset]),
                    });
                    apply_row(&mut raw, &mut compressed, slot, 10_000 + row as i32);
                }

                for row in (accepted..ROWS).rev() {
                    let entry = undo[row];
                    raw[entry.raw_offset] = entry.raw;
                    if let Some(offset) = entry.compressed_offset {
                        compressed[offset] = entry.compressed;
                    }
                }

                let mut expected_raw = initial_raw;
                let mut expected_compressed = initial_compressed;
                for row in 0..accepted {
                    apply_row(
                        &mut expected_raw,
                        &mut expected_compressed,
                        start + row,
                        10_000 + row as i32,
                    );
                }
                assert_eq!(raw, expected_raw, "start={start} accepted={accepted}");
                assert_eq!(
                    compressed, expected_compressed,
                    "start={start} accepted={accepted}"
                );
            }
        }
    }
}
