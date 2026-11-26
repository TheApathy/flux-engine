// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence stop ids, including the vLLM/SGLang `ignore_eos` extension.

/// Stop ids for one sequence: the model EOS ids merged with the request's own
/// stop tokens. An `ignore_eos` request keeps only its own stop tokens, so the
/// model EOS ids are sampled and counted like any other token and the turn runs
/// to `max_tokens`, a stop string or a requested stop token (vLLM semantics).
pub(super) fn sequence_stop_ids(
    model_eos: &[u32],
    stop_tokens: Vec<u32>,
    ignore_eos: bool,
) -> Vec<u32> {
    if !ignore_eos && stop_tokens.is_empty() {
        return model_eos.to_vec();
    }
    let mut ids = if ignore_eos {
        Vec::new()
    } else {
        model_eos.to_vec()
    };
    ids.extend(stop_tokens);
    ids.sort_unstable();
    ids.dedup();
    ids
}

#[cfg(test)]
mod tests {
    use super::sequence_stop_ids;

    const IM_END: u32 = 30;
    const END_OF_TEXT: u32 = 20;
    const IM_START: u32 = 10;
    const MODEL_EOS: [u32; 3] = [IM_END, END_OF_TEXT, IM_START];

    #[test]
    fn model_eos_ids_stop_by_default() {
        assert_eq!(
            sequence_stop_ids(&MODEL_EOS, Vec::new(), false),
            MODEL_EOS.to_vec()
        );
        assert_eq!(
            sequence_stop_ids(&MODEL_EOS, vec![7, IM_END], false),
            vec![7, IM_START, END_OF_TEXT, IM_END]
        );
    }

    #[test]
    fn ignore_eos_keeps_only_the_requested_stop_tokens() {
        assert!(sequence_stop_ids(&MODEL_EOS, Vec::new(), true).is_empty());
        assert_eq!(
            sequence_stop_ids(&MODEL_EOS, vec![9, 7, 9], true),
            vec![7, 9]
        );
    }
}
