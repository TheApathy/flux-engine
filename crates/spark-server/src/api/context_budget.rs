// SPDX-License-Identifier: AGPL-3.0-only

//! Checked admission for the shared input-plus-output context budget.

pub(super) fn admitted_total(
    prompt_tokens: usize,
    requested_output_tokens: usize,
    max_seq_len: usize,
) -> Option<usize> {
    prompt_tokens
        .checked_add(requested_output_tokens)
        .filter(|&total| total <= max_seq_len)
}

/// A server-defaulted output allowance (the client sent no max-tokens field)
/// shrinks to the context left after the prompt, as vLLM does, so a large
/// default never turns a long prompt into a 400. With no room left the
/// default is kept, so admission still reports the overflow.
pub(super) fn defaulted_output_allowance(
    default_tokens: usize,
    prompt_tokens: usize,
    max_seq_len: usize,
) -> usize {
    match max_seq_len.saturating_sub(prompt_tokens) {
        0 => default_tokens,
        room => default_tokens.min(room),
    }
}

#[cfg(test)]
mod defaulted_output_allowance_tests {
    use super::{admitted_total, defaulted_output_allowance};

    #[test]
    fn default_fits_unchanged_when_there_is_room() {
        assert_eq!(defaulted_output_allowance(32768, 1000, 262_144), 32768);
        assert_eq!(defaulted_output_allowance(32768, 0, 32768), 32768);
    }

    #[test]
    fn default_shrinks_to_the_remaining_context_and_is_admitted() {
        let allowance = defaulted_output_allowance(32768, 6000, 8192);
        assert_eq!(allowance, 2192);
        assert_eq!(admitted_total(6000, allowance, 8192), Some(8192));
    }

    #[test]
    fn a_full_context_still_fails_admission() {
        let allowance = defaulted_output_allowance(32768, 8192, 8192);
        assert_eq!(admitted_total(8192, allowance, 8192), None);
        let allowance = defaulted_output_allowance(32768, 9000, 8192);
        assert_eq!(admitted_total(9000, allowance, 8192), None);
    }
}

#[cfg(test)]
mod tests {
    use super::admitted_total;

    #[test]
    fn million_token_boundary_is_inclusive() {
        assert_eq!(admitted_total(999_999, 1, 1_000_000), Some(1_000_000));
        assert_eq!(admitted_total(999_999, 2, 1_000_000), None);
    }

    #[test]
    fn addition_overflow_is_rejected() {
        assert_eq!(admitted_total(usize::MAX, 1, usize::MAX), None);
    }

    #[test]
    fn native_262k_chat_boundary_is_exact() {
        assert_eq!(admitted_total(262_143, 1, 262_144), Some(262_144));
        assert_eq!(admitted_total(262_143, 2, 262_144), None);
        assert_eq!(admitted_total(usize::MAX, 2, 262_144), None);
    }

    #[test]
    fn chat_checks_effective_sampling_budget_before_dispatch() {
        let source = include_str!("chat/mod.rs");
        let build = source.find("sampling_setup::build_sampling(").unwrap();
        // Admission (`admit_request_output` -> `request_contract::admit_output`
        // -> `admitted_total`) runs on the effective budget, and the decode
        // deadline is derived from the post-admission (possibly clamped)
        // `max_tokens`, both before dispatch.
        let budget = source[build..]
            .find("admit_request_output(&state, prompt_len, max_tokens)")
            .unwrap()
            + build;
        let deadline = source[build..]
            .find("sampling_setup::contract_decode_deadline(")
            .unwrap()
            + build;
        let dispatch = source[build..].find("if req.stream").unwrap() + build;
        assert!(build < budget && budget < deadline && deadline < dispatch);
        let contract = include_str!("request_contract.rs");
        assert!(contract.contains(
            "super::context_budget::admitted_total(prompt_len, max_tokens, max_seq_len)"
        ));
    }

    #[test]
    fn yarn_images_are_rejected_before_message_preprocessing() {
        // The YaRN image rejection lives in `image_admission::validate`, which
        // takes `state.yarn_context` and runs before message preprocessing.
        let chat = include_str!("chat/mod.rs");
        let admission = include_str!("chat/image_admission.rs");
        let reject = chat.find("image_admission::validate(").unwrap();
        let yarn_arg = chat[reject..].find("state.yarn_context").unwrap() + reject;
        let preprocess = chat.find("msg_entry::build_msg_entries").unwrap();
        assert!(reject < yarn_arg && yarn_arg < preprocess);
        assert!(admission.contains("if yarn_context"));
        assert!(admission.contains("unsupported_multimodal_context"));
    }
}
