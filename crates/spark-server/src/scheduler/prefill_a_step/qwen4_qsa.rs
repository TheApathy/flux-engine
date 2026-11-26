// SPDX-License-Identifier: AGPL-3.0-only

const DENSE_LIMIT: usize = 2048;

/// Returns `true` when the request's immutable token budget (`prompt_tokens +
/// max_tokens`) can reach the Qwen4 sparse QSA threshold (`DENSE_LIMIT`).
///
/// Requests strictly below `DENSE_LIMIT` skip QSA side-cache indexing; their
/// cached prefix nodes remain tagged `qsa_populated = false` in `PrefixCache`
/// so a subsequent `>= DENSE_LIMIT` turn never reuses unpopulated QSA pages.
#[inline]
pub(super) fn request_can_reach_qwen4_qsa(prompt_tokens: usize, max_tokens: usize) -> bool {
    prompt_tokens.saturating_add(max_tokens) >= DENSE_LIMIT
}

#[cfg(test)]
mod tests {
    use super::request_can_reach_qwen4_qsa;

    #[test]
    fn skips_only_requests_strictly_below_dense_limit() {
        assert!(!request_can_reach_qwen4_qsa(38, 400));
        assert!(!request_can_reach_qwen4_qsa(2047, 0));
        assert!(request_can_reach_qwen4_qsa(2047, 1));
        assert!(request_can_reach_qwen4_qsa(21, 2048));
    }

    #[test]
    fn overflow_is_conservative() {
        assert!(request_can_reach_qwen4_qsa(usize::MAX, 1));
    }
}
