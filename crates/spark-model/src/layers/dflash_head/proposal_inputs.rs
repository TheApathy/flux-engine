// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side encoding for fixed-address DFlash proposal inputs.

use anyhow::{Context, Result, bail};

#[cfg(test)]
#[path = "proposal_position_tests.rs"]
mod position_tests;

/// Encode chronological context positions followed by noise-row positions
/// directly into the reusable pinned H2D buffer.
pub(super) fn encode_position_ids(
    dst: &mut [u8],
    position: usize,
    eff_ctx: usize,
    noise_rows: usize,
) -> Result<usize> {
    let row_count = eff_ctx
        .checked_add(noise_rows)
        .context("DFlash position row count overflow")?;
    let byte_count = row_count
        .checked_mul(size_of::<i32>())
        .context("DFlash position byte count overflow")?;
    if dst.len() < byte_count {
        bail!(
            "DFlash position input needs {byte_count} bytes, pinned buffer has {}",
            dst.len()
        );
    }

    let ctx_start = position
        .checked_sub(eff_ctx)
        .context("DFlash context extends before physical position zero")?;
    // Validate the last positions BEFORE touching pinned staging. A failed
    // upload must not leave a partially rewritten request visible to DMA.
    if eff_ctx > 0 {
        i32::try_from(position - 1).context("DFlash context exceeds i32 kernel ABI")?;
    }
    if noise_rows > 0 {
        let last = position
            .checked_add(noise_rows - 1)
            .context("DFlash noise position overflow")?;
        i32::try_from(last).context("DFlash noise position exceeds i32 kernel ABI")?;
    }
    for (row, bytes) in dst[..byte_count].chunks_exact_mut(4).enumerate() {
        let absolute = if row < eff_ctx {
            ctx_start
                .checked_add(row)
                .context("DFlash context position overflow")?
        } else {
            position
                .checked_add(row - eff_ctx)
                .context("DFlash noise position overflow")?
        };
        let encoded = i32::try_from(absolute)
            .with_context(|| format!("DFlash position {absolute} exceeds i32 kernel ABI"))?;
        bytes.copy_from_slice(&encoded.to_le_bytes());
    }
    Ok(byte_count)
}

/// Encode `[context zeros, anchor, feedback-or-mask rows]` directly into a
/// reusable pinned input buffer.
pub(super) fn encode_noise_token_ids(
    dst: &mut [u8],
    eff_ctx: usize,
    last_token: u32,
    feedback_rows: usize,
    committed: &[Option<u32>],
    mask_id: u32,
) -> Result<usize> {
    if !committed.is_empty() && committed.len() < feedback_rows {
        bail!(
            "DFlash feedback needs {feedback_rows} committed slots, got {}",
            committed.len()
        );
    }
    let row_count = eff_ctx
        .checked_add(1)
        .and_then(|rows| rows.checked_add(feedback_rows))
        .context("DFlash noise token row count overflow")?;
    let byte_count = row_count
        .checked_mul(size_of::<i32>())
        .context("DFlash noise token byte count overflow")?;
    if dst.len() < byte_count {
        bail!(
            "DFlash noise token input needs {byte_count} bytes, pinned buffer has {}",
            dst.len()
        );
    }
    let last_token = i32::try_from(last_token).context("DFlash anchor token exceeds i32 ABI")?;
    let mask_id = i32::try_from(mask_id).context("DFlash mask token exceeds i32 ABI")?;

    for (row, bytes) in dst[..byte_count].chunks_exact_mut(4).enumerate() {
        let token = if row < eff_ctx {
            0
        } else if row == eff_ctx {
            last_token
        } else {
            match committed.get(row - eff_ctx - 1).copied().flatten() {
                Some(token) => {
                    i32::try_from(token).context("DFlash committed token exceeds i32 ABI")?
                }
                None => mask_id,
            }
        };
        bytes.copy_from_slice(&token.to_le_bytes());
    }
    Ok(byte_count)
}

/// Decode exactly `token_count` little-endian u32 draft IDs from reusable
/// host staging. Trailing capacity bytes are deliberately ignored.
pub(super) fn decode_token_ids(src: &[u8], token_count: usize) -> Result<Vec<u32>> {
    let byte_count = token_count
        .checked_mul(size_of::<u32>())
        .context("DFlash draft output byte count overflow")?;
    if src.len() < byte_count {
        bail!(
            "DFlash draft output needs {byte_count} bytes, staging buffer has {}",
            src.len()
        );
    }
    Ok(src[..byte_count]
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

/// Clear any stale frame before collection and publish the replacement only
/// after every fallible readback/decoding step succeeds. This makes failure
/// ordering explicit at the producer boundary instead of relying on callers
/// to remember to clear partially collected state.
pub(super) fn publish_after_success<T, Frame>(
    pending: &mut Option<Frame>,
    collect: impl FnOnce() -> Result<(T, Frame)>,
) -> Result<T> {
    *pending = None;
    let (value, frame) = collect()?;
    *pending = Some(frame);
    Ok(value)
}

/// Encode deterministic selector uniforms directly into reusable pinned H2D
/// staging. Validation happens before the first byte is touched.
pub(super) fn encode_selector_uniforms(dst: &mut [u8], uniforms: &[f32]) -> Result<usize> {
    let byte_count = uniforms
        .len()
        .checked_mul(size_of::<f32>())
        .context("DFlash selector uniform byte count overflow")?;
    if dst.len() < byte_count {
        bail!(
            "DFlash selector uniforms need {byte_count} bytes, pinned buffer has {}",
            dst.len()
        );
    }
    if uniforms
        .iter()
        .any(|uniform| !uniform.is_finite() || !(0.0..1.0).contains(uniform))
    {
        bail!("DFlash selector uniform must be finite and in [0, 1)");
    }
    for (&uniform, bytes) in uniforms
        .iter()
        .zip(dst[..byte_count].chunks_exact_mut(size_of::<f32>()))
    {
        bytes.copy_from_slice(&uniform.to_le_bytes());
    }
    Ok(byte_count)
}

#[cfg(test)]
mod tests {
    use super::{
        decode_token_ids, encode_noise_token_ids, encode_position_ids, encode_selector_uniforms,
        publish_after_success,
    };

    fn decode(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn context_then_noise_positions_are_chronological() {
        let mut dst = [0xAA; 32];
        let used = encode_position_ids(&mut dst, 10, 4, 3).unwrap();
        assert_eq!(used, 28);
        assert_eq!(decode(&dst[..used]), [6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(&dst[used..], &[0xAA; 4]);
    }

    #[test]
    fn short_prompts_reject_context_before_physical_origin() {
        let mut dst = [0xA5; 24];
        assert!(encode_position_ids(&mut dst, 2, 4, 2).is_err());
        assert_eq!(dst, [0xA5; 24]);
    }

    #[test]
    fn one_million_context_fits_the_kernel_abi() {
        let mut dst = [0; 16];
        let used = encode_position_ids(&mut dst, 1_048_575, 2, 2).unwrap();
        assert_eq!(
            decode(&dst[..used]),
            [1_048_573, 1_048_574, 1_048_575, 1_048_576]
        );
    }

    #[test]
    fn insufficient_storage_and_i32_overflow_fail_closed() {
        assert!(encode_position_ids(&mut [0; 7], 0, 1, 1).is_err());
        assert!(encode_position_ids(&mut [0; 4], i32::MAX as usize + 1, 0, 1).is_err());
    }

    #[test]
    fn noise_tokens_encode_context_anchor_and_feedback_without_padding_writes() {
        let mut dst = [0xAA; 32];
        let used = encode_noise_token_ids(&mut dst, 2, 42, 3, &[None, Some(7), None], 99).unwrap();
        assert_eq!(used, 24);
        assert_eq!(decode(&dst[..used]), [0, 0, 42, 99, 7, 99]);
        assert_eq!(&dst[used..], &[0xAA; 8]);
    }

    #[test]
    fn empty_feedback_state_encodes_the_single_pass_all_mask_layout() {
        let mut dst = [0; 20];
        let used = encode_noise_token_ids(&mut dst, 1, 42, 3, &[], 99).unwrap();
        assert_eq!(used, 20);
        assert_eq!(decode(&dst), [0, 42, 99, 99, 99]);
    }

    #[test]
    fn draft_output_decodes_directly_from_reusable_staging() {
        let bytes = [1, 0, 0, 0, 0x78, 0x56, 0x34, 0x12, 0xAA];
        assert_eq!(decode_token_ids(&bytes, 2).unwrap(), [1, 0x1234_5678]);
        assert!(decode_token_ids(&bytes[..7], 2).is_err());
    }

    #[test]
    fn noise_token_shape_and_token_overflow_fail_closed() {
        assert!(encode_noise_token_ids(&mut [0; 8], 1, 42, 1, &[None], 99).is_err());
        assert!(encode_noise_token_ids(&mut [0; 4], 0, i32::MAX as u32 + 1, 0, &[], 99).is_err());
        assert!(encode_noise_token_ids(&mut [0; 12], 0, 42, 2, &[None], 99).is_err());
    }

    #[test]
    fn selector_uniforms_are_exact_and_fail_before_partial_write() {
        let mut dst = [0xAA; 12];
        let used = encode_selector_uniforms(&mut dst, &[0.0, 0.5]).unwrap();
        assert_eq!(used, 8);
        assert_eq!(f32::from_le_bytes(dst[0..4].try_into().unwrap()), 0.0);
        assert_eq!(f32::from_le_bytes(dst[4..8].try_into().unwrap()), 0.5);
        assert_eq!(&dst[8..], &[0xAA; 4]);

        let mut untouched = [0xA5; 8];
        assert!(encode_selector_uniforms(&mut untouched, &[0.5, 1.0]).is_err());
        assert_eq!(untouched, [0xA5; 8]);
    }

    #[test]
    fn sampled_frame_publication_is_atomic_across_each_fallible_stage() {
        for (failure, label) in ["pair", "q", "decode"].into_iter().enumerate() {
            let mut pending = Some(0xA5u8);
            let result: anyhow::Result<u32> = publish_after_success(&mut pending, || {
                if failure == 0 {
                    anyhow::bail!("pair readback failed")
                }
                if failure == 1 {
                    anyhow::bail!("q readback failed")
                }
                if failure == 2 {
                    anyhow::bail!("decode failed")
                }
                Ok((7, 0x5Au8))
            });
            assert!(result.is_err(), "{label}");
            assert_eq!(pending, None, "{label}");
        }

        let mut pending = Some(0xA5u8);
        let value = publish_after_success(&mut pending, || Ok((7u32, 0x5Au8))).unwrap();
        assert_eq!(value, 7);
        assert_eq!(pending, Some(0x5A));
    }
}
