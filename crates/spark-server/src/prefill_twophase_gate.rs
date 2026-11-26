// SPDX-License-Identifier: AGPL-3.0-only

/// Decide whether a request may enter the non-replayable two-phase prefill
/// path. The model capability is checked while the sequence is still pristine.
pub const fn should_use_twophase(
    supports_twophase: bool,
    chunk_offset: usize,
    prompt_len: usize,
    max_prefill_tokens: usize,
) -> bool {
    supports_twophase && chunk_offset == 0 && prompt_len > max_prefill_tokens
}

#[cfg(test)]
mod tests {
    use super::should_use_twophase;

    #[test]
    fn selection_requires_model_capability_and_pristine_long_prompt() {
        assert!(should_use_twophase(true, 0, 32_768, 16_384));
        assert!(!should_use_twophase(false, 0, 32_768, 16_384));
        assert!(!should_use_twophase(true, 1, 32_768, 16_384));
        assert!(!should_use_twophase(true, 0, 16_384, 16_384));
        assert!(!should_use_twophase(true, 0, 8_192, 16_384));
    }
}
