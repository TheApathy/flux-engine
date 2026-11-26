// SPDX-License-Identifier: AGPL-3.0-only

//! Target MRoPE belongs to the native head, not the qwen3 DFlash2 drafter.

use crate::model::types::TransformerModel;
use crate::speculative::ProposerState;
use crate::traits::{RotaryPositions, SequenceState};
use anyhow::{Result, ensure};

pub(super) fn sync_proposer_state_rotary(
    state: &mut dyn ProposerState,
    rotary_positions: &RotaryPositions,
) -> Result<()> {
    if let Some(mtp) = state
        .as_any_mut()
        .downcast_mut::<crate::layers::mtp_head::MtpProposerState>()
    {
        mtp.rotary_positions = rotary_positions.clone();
    } else if let Some(mtp) = state
        .as_any_mut()
        .downcast_mut::<crate::layers::qwen4_mtp::Qwen4MtpState>()
    {
        mtp.rotary_positions = rotary_positions.clone();
    } else if state.as_any().is::<crate::layers::DflashProposerState>() {
        // Trained qwen3/default-RoPE DFlash2 uses chronological PHYSICAL
        // context and noise indices. Do not change any cached/ring index.
    } else {
        ensure!(
            rotary_positions.is_identity(),
            "this proposer does not admit target image rotary state"
        );
    }
    Ok(())
}

impl TransformerModel {
    pub(super) fn sync_proposer_rotary(&self, seq: &mut SequenceState) -> Result<()> {
        let Some(state) = seq.proposer_state.as_mut() else {
            return Ok(());
        };
        sync_proposer_state_rotary(state.as_mut(), &seq.rotary_positions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::mtp_head::MtpProposerState;
    use crate::traits::ImageSpan;
    use std::any::Any;

    struct UnsupportedState;

    impl ProposerState for UnsupportedState {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    fn image_positions() -> RotaryPositions {
        RotaryPositions::from_image_spans(
            6,
            32,
            &[ImageSpan {
                start: 1,
                height: 2,
                width: 2,
            }],
        )
        .unwrap()
    }

    #[test]
    fn explicit_state_sync_copies_target_map() {
        let target = image_positions();
        let mut state = MtpProposerState {
            rotary_positions: RotaryPositions::identity(),
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
        };
        sync_proposer_state_rotary(&mut state, &target).unwrap();
        assert_eq!(state.rotary_positions, target);
    }

    #[test]
    fn unsupported_state_rejects_images_but_preserves_text_contract() {
        let mut state = UnsupportedState;
        sync_proposer_state_rotary(&mut state, &RotaryPositions::identity()).unwrap();
        assert!(sync_proposer_state_rotary(&mut state, &image_positions()).is_err());
    }
}
