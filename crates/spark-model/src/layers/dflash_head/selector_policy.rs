// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed-address staging for sampled DFlash2 q-policy modifiers.

use anyhow::{Context, Result, bail, ensure};

use crate::dflash_sampling::DFLASH2_MAX_Q_MODIFIERS;

pub(super) const SELECTOR_Q_MODIFIER_VALUES_OFFSET: usize =
    DFLASH2_MAX_Q_MODIFIERS * size_of::<u32>();
pub(super) const SELECTOR_Q_MODIFIER_BYTES: usize =
    SELECTOR_Q_MODIFIER_VALUES_OFFSET + DFLASH2_MAX_Q_MODIFIERS * size_of::<f32>();

/// Encode ordered finite request biases followed by target hard masks.
///
/// The CUDA kernel interprets negative infinity only in this internal staging
/// ABI as a hard mask. Public request biases must remain finite. Validation is
/// completed before `dst` is touched so an invalid next request cannot expose
/// a partially rewritten DMA buffer.
pub(super) fn encode_selector_q_modifiers(
    dst: &mut [u8],
    logit_bias: &[(u32, f32)],
    forbidden_token_ids: &[u32],
) -> Result<usize> {
    let count = logit_bias
        .len()
        .checked_add(forbidden_token_ids.len())
        .context("DFlash q modifier count overflow")?;
    if count > DFLASH2_MAX_Q_MODIFIERS {
        bail!(
            "DFlash q modifier count {count} exceeds bounded ABI capacity {DFLASH2_MAX_Q_MODIFIERS}"
        );
    }
    ensure!(
        logit_bias.iter().all(|(_, bias)| bias.is_finite()),
        "DFlash q logit bias contains a non-finite value"
    );
    ensure!(
        forbidden_token_ids.windows(2).all(|pair| pair[0] < pair[1]),
        "DFlash q hard-mask token ids must be sorted and unique"
    );
    if dst.len() < SELECTOR_Q_MODIFIER_BYTES {
        bail!(
            "DFlash q modifier staging needs {SELECTOR_Q_MODIFIER_BYTES} bytes, pinned buffer has {}",
            dst.len()
        );
    }

    let mut encoded = [0u8; SELECTOR_Q_MODIFIER_BYTES];
    for (index, &(token, bias)) in logit_bias.iter().enumerate() {
        let id_start = index * size_of::<u32>();
        encoded[id_start..id_start + size_of::<u32>()].copy_from_slice(&token.to_le_bytes());
        let value_start = SELECTOR_Q_MODIFIER_VALUES_OFFSET + index * size_of::<f32>();
        encoded[value_start..value_start + size_of::<f32>()].copy_from_slice(&bias.to_le_bytes());
    }
    for (offset, &token) in forbidden_token_ids.iter().enumerate() {
        let index = logit_bias.len() + offset;
        let id_start = index * size_of::<u32>();
        encoded[id_start..id_start + size_of::<u32>()].copy_from_slice(&token.to_le_bytes());
        let value_start = SELECTOR_Q_MODIFIER_VALUES_OFFSET + index * size_of::<f32>();
        encoded[value_start..value_start + size_of::<f32>()]
            .copy_from_slice(&f32::NEG_INFINITY.to_le_bytes());
    }
    dst[..SELECTOR_Q_MODIFIER_BYTES].copy_from_slice(&encoded);
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_u32(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn decode_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn ordered_biases_and_hard_masks_have_an_exact_fixed_layout() {
        let mut dst = [0xA5; SELECTOR_Q_MODIFIER_BYTES];
        let count =
            encode_selector_q_modifiers(&mut dst, &[(9, -100.0), (9, 0.25), (3, 1.5)], &[7, 11])
                .unwrap();
        assert_eq!(count, 5);
        let ids = decode_u32(&dst[..SELECTOR_Q_MODIFIER_VALUES_OFFSET]);
        let values = decode_f32(&dst[SELECTOR_Q_MODIFIER_VALUES_OFFSET..]);
        assert_eq!(&ids[..count], &[9, 9, 3, 7, 11]);
        assert_eq!(&values[..3], &[-100.0, 0.25, 1.5]);
        assert!(values[3].is_infinite() && values[3].is_sign_negative());
        assert!(values[4].is_infinite() && values[4].is_sign_negative());
        assert!(ids[count..].iter().all(|&id| id == 0));
        assert!(values[count..].iter().all(|&value| value == 0.0));
    }

    #[test]
    fn invalid_or_oversized_policy_fails_before_mutating_staging() {
        let original = [0xA5; SELECTOR_Q_MODIFIER_BYTES];
        for (biases, masks) in [
            (vec![(1, f32::NAN)], vec![]),
            (vec![], vec![2, 1]),
            (
                (0..=DFLASH2_MAX_Q_MODIFIERS)
                    .map(|token| (token as u32, 0.0))
                    .collect(),
                vec![],
            ),
        ] {
            let mut dst = original;
            assert!(encode_selector_q_modifiers(&mut dst, &biases, &masks).is_err());
            assert_eq!(dst, original);
        }
        assert!(encode_selector_q_modifiers(&mut [0; 8], &[], &[]).is_err());
    }
}
