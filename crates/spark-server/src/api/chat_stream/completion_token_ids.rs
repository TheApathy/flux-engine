// SPDX-License-Identifier: AGPL-3.0-only

//! Terminal framing for scheduler-authoritative completion token IDs.

use crate::openai::{ChatCompletionChunk, Usage};

pub(super) fn validate(
    return_token_ids: bool,
    completion_tokens: usize,
    completion_token_ids: &[u32],
) -> Result<(), String> {
    if return_token_ids && completion_token_ids.len() != completion_tokens {
        return Err(format!(
            "completion token-ID cardinality {} != usage {}",
            completion_token_ids.len(),
            completion_tokens
        ));
    }
    Ok(())
}

pub(super) fn validate_or_cancel(
    cancel: &std::sync::atomic::AtomicBool,
    return_token_ids: bool,
    completion_tokens: usize,
    completion_token_ids: &[u32],
) -> Result<(), String> {
    validate(return_token_ids, completion_tokens, completion_token_ids)
        .inspect_err(|_| cancel.store(true, std::sync::atomic::Ordering::Release))
}

/// Build the terminal chunk sequence. The scheduler snapshot is returned only
/// for opted-in requests and only on the finish-bearing choice.
pub(super) fn terminal_chunks(
    model: &str,
    id: &str,
    finish_reason: &str,
    usage: Usage,
    include_usage: bool,
    return_token_ids: bool,
    completion_token_ids: Vec<u32>,
    interventions: Option<crate::openai::InterventionLedger>,
    rendered_prompt_evidence: Option<crate::openai::RenderedPromptEvidence>,
    flux_stop_reason: Option<&'static str>,
) -> Result<Vec<ChatCompletionChunk>, String> {
    validate(
        return_token_ids,
        usage.completion_tokens,
        &completion_token_ids,
    )?;
    let final_ids = if return_token_ids {
        completion_token_ids
    } else {
        Vec::new()
    };

    if include_usage {
        Ok(vec![
            ChatCompletionChunk::final_chunk_no_usage(model, id, finish_reason)
                .with_token_ids(final_ids)
                .with_flux_stop_reason(flux_stop_reason)
                .with_interventions(interventions)
                .with_rendered_prompt_evidence(rendered_prompt_evidence),
            ChatCompletionChunk::usage_only_chunk(model, id, usage),
        ])
    } else {
        Ok(vec![
            ChatCompletionChunk::done_chunk(model, id, finish_reason, usage)
                .with_token_ids(final_ids)
                .with_flux_stop_reason(flux_stop_reason)
                .with_interventions(interventions)
                .with_rendered_prompt_evidence(rendered_prompt_evidence),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(completion_tokens: usize) -> Usage {
        Usage {
            prompt_tokens: 3,
            completion_tokens,
            total_tokens: completion_tokens + 3,
            prompt_tokens_details: None,
            completion_tokens_details: None,
            time_to_first_token_ms: 1.0,
            response_tokens_per_second: 2.0,
        }
    }

    fn json(chunks: &[ChatCompletionChunk]) -> Vec<serde_json::Value> {
        chunks
            .iter()
            .map(|chunk| serde_json::to_value(chunk).unwrap())
            .collect()
    }

    #[test]
    fn opted_in_ids_appear_once_on_the_finish_choice_for_both_usage_layouts() {
        for include_usage in [false, true] {
            let values = json(
                &terminal_chunks(
                    "m",
                    "id",
                    "stop",
                    usage(4),
                    include_usage,
                    true,
                    vec![7, 11, 13, 17],
                    None,
                    None,
                    None,
                )
                .unwrap(),
            );
            let ids: Vec<&serde_json::Value> = values
                .iter()
                .flat_map(|chunk| chunk["choices"].as_array().into_iter().flatten())
                .filter_map(|choice| choice.get("token_ids"))
                .collect();
            assert_eq!(ids, vec![&serde_json::json!([7, 11, 13, 17])]);
            assert_eq!(
                values
                    .iter()
                    .filter_map(|chunk| chunk["usage"]["completion_tokens"].as_u64())
                    .collect::<Vec<_>>(),
                vec![4]
            );
            assert_eq!(
                values
                    .iter()
                    .filter_map(|chunk| chunk["choices"][0]["finish_reason"].as_str())
                    .collect::<Vec<_>>(),
                vec!["stop"]
            );
        }
    }

    #[test]
    fn usage_layout_keeps_finish_before_the_choices_empty_usage_frame() {
        let values = json(
            &terminal_chunks(
                "m",
                "id",
                "stop",
                usage(2),
                true,
                true,
                vec![7, 11],
                None,
                None,
                None,
            )
            .unwrap(),
        );
        assert_eq!(values.len(), 2);
        assert_eq!(values[0]["choices"][0]["finish_reason"], "stop");
        assert_eq!(
            values[0]["choices"][0]["token_ids"],
            serde_json::json!([7, 11])
        );
        assert_eq!(values[1]["choices"], serde_json::json!([]));
        assert_eq!(values[1]["usage"]["completion_tokens"], 2);
    }

    #[test]
    fn opt_out_omits_token_ids_and_keeps_usage_unchanged() {
        for include_usage in [false, true] {
            let values = json(
                &terminal_chunks(
                    "m",
                    "id",
                    "length",
                    usage(2),
                    include_usage,
                    false,
                    vec![19, 23],
                    None,
                    None,
                    None,
                )
                .unwrap(),
            );
            assert!(
                !serde_json::to_string(&values)
                    .unwrap()
                    .contains("token_ids")
            );
            assert!(
                values
                    .iter()
                    .any(|chunk| chunk["usage"]["completion_tokens"] == 2)
            );
        }
    }

    #[test]
    fn opted_in_cardinality_mismatch_refuses_all_terminal_chunks() {
        let error = terminal_chunks(
            "m",
            "id",
            "stop",
            usage(2),
            true,
            true,
            vec![29],
            None,
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(error, "completion token-ID cardinality 1 != usage 2");

        let cancel = std::sync::atomic::AtomicBool::new(false);
        let error = validate_or_cancel(&cancel, true, 2, &[29]).unwrap_err();
        assert_eq!(error, "completion token-ID cardinality 1 != usage 2");
        assert!(cancel.load(std::sync::atomic::Ordering::Acquire));
    }

    #[test]
    fn intervention_ledger_is_terminal_and_opt_in_only() {
        for include_usage in [false, true] {
            let ledger = crate::openai::InterventionLedger::default();
            let values = json(
                &terminal_chunks(
                    "m",
                    "id",
                    "stop",
                    usage(1),
                    include_usage,
                    false,
                    vec![7],
                    Some(ledger),
                    None,
                    None,
                )
                .unwrap(),
            );
            assert_eq!(values[0]["choices"][0]["finish_reason"], "stop");
            assert_eq!(
                values[0]["interventions"]["schema"],
                "atlas.interventions.v1"
            );
            assert_eq!(values[0]["interventions"]["policy"], "report_only");
            assert_eq!(values[0]["interventions"]["events"], serde_json::json!([]));
            assert!(
                values
                    .iter()
                    .skip(1)
                    .all(|value| value.get("interventions").is_none())
            );
        }
    }

    #[test]
    fn rendered_prompt_evidence_is_response_bound_and_terminal_only() {
        let evidence = crate::openai::RenderedPromptEvidence::from_digest(
            "11111111111111111111111111111111".into(),
            "id".into(),
            "f5b85e4d259cf2d0ebba85f37228213399af7362db9f4f307b8dd6830b983de1".into(),
        );
        let values = json(
            &terminal_chunks(
                "m",
                "id",
                "stop",
                usage(1),
                true,
                true,
                vec![7],
                Some(crate::openai::InterventionLedger::default()),
                Some(evidence),
                None,
            )
            .unwrap(),
        );
        assert_eq!(values[0]["rendered_prompt_evidence"]["response_id"], "id");
        assert_eq!(
            values[0]["rendered_prompt_evidence"]["nonce"],
            "11111111111111111111111111111111"
        );
        assert_eq!(
            values[0]["rendered_prompt_evidence"]["prompt_sha256"],
            "f5b85e4d259cf2d0ebba85f37228213399af7362db9f4f307b8dd6830b983de1"
        );
        let evidence = values[0]["rendered_prompt_evidence"].as_object().unwrap();
        assert_eq!(evidence.len(), 4);
        assert!(evidence.get("prompt_bytes").is_none());
        assert!(evidence.get("prompt_utf8").is_none());
        assert!(evidence.get("rendered_prompt_utf8").is_none());
        assert!(values[1].get("rendered_prompt_evidence").is_none());
    }

    #[test]
    fn dsv_and_completions_done_arms_ignore_the_generic_snapshot() {
        let chat = include_str!("mod.rs");
        let dsv = chat
            .split_once("if state.dsv41")
            .unwrap()
            .1
            .split_once("let mut stream_state")
            .unwrap()
            .0;
        assert!(dsv.contains("StreamEvent::Done"));
        assert!(dsv.contains(".."));
        assert!(!dsv.contains("completion_token_ids"));

        let completions = include_str!("../completions.rs")
            .split_once("pub(super) async fn completions_stream")
            .unwrap()
            .1
            .split_once("\n#[cfg(test)]")
            .unwrap()
            .0;
        assert!(completions.contains("StreamEvent::Done"));
        assert!(completions.contains(".."));
        assert!(!completions.contains("completion_token_ids"));
    }

    #[test]
    fn a_guard_stop_names_its_cause_on_the_finish_choice_only() {
        for include_usage in [false, true] {
            let values = json(
                &terminal_chunks(
                    "m",
                    "id",
                    "stop",
                    usage(3),
                    include_usage,
                    false,
                    vec![7, 11, 13],
                    None,
                    None,
                    Some("catastrophic_guard"),
                )
                .unwrap(),
            );
            assert_eq!(values[0]["choices"][0]["finish_reason"], "stop");
            let finish = &values[0]["choices"][0];
            assert_eq!(finish["flux_stop_reason"], "catastrophic_guard");
            if include_usage {
                assert!(values[1].get("choices").and_then(|c| c.get(0)).is_none());
            }
        }
        let plain = json(
            &terminal_chunks(
                "m",
                "id",
                "length",
                usage(3),
                false,
                false,
                vec![7, 11, 13],
                None,
                None,
                None,
            )
            .unwrap(),
        );
        let finish = &plain[0]["choices"][0];
        assert!(
            finish.get("flux_stop_reason").is_none(),
            "omitted when unset"
        );
    }
}
