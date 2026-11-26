// SPDX-License-Identifier: AGPL-3.0-only

//! OpenAI logprob publication for streaming chat tokens.
//!
//! Text decoding is not one-token-in/one-delta-out: a streaming decoder can
//! buffer bytes, and policy filters can intentionally suppress a token's text.
//! Logprobs still describe committed scheduler tokens, so every
//! `TokenWithLogprobs` event needs exactly one choice entry even when no text
//! chunk was produced. In that case we use an empty-delta progress chunk.

use crate::api::TokenLogprobs;
use crate::openai::{ChatCompletionChunk, ChoiceLogprobs, TokenLogprobInfo, TopLogprob};
use crate::tokenizer::ChatTokenizer;

use super::ChunkVec;

pub(super) fn attach_token_logprobs(
    chunks: ChunkVec,
    tokenizer: &ChatTokenizer,
    model: &str,
    id: &str,
    logprobs: TokenLogprobs,
) -> ChunkVec {
    let sampled = TokenLogprobInfo {
        token: tokenizer.decode(&[logprobs.token_id]).unwrap_or_default(),
        logprob: logprobs.logprob,
        bytes: None,
        top_logprobs: logprobs
            .top
            .into_iter()
            .map(|(token_id, logprob)| TopLogprob {
                token: tokenizer.decode(&[token_id]).unwrap_or_default(),
                logprob,
                bytes: None,
            })
            .collect(),
    };
    attach_choice_logprobs(chunks, model, id, sampled)
}

fn attach_choice_logprobs(
    mut chunks: ChunkVec,
    model: &str,
    id: &str,
    sampled: TokenLogprobInfo,
) -> ChunkVec {
    let carrier = chunks.iter().position(|chunk| !chunk.choices.is_empty());
    let carrier = carrier.unwrap_or_else(|| {
        chunks.push(ChatCompletionChunk::progress_chunk(model, id));
        chunks.len() - 1
    });
    let choice = chunks[carrier]
        .choices
        .first_mut()
        .expect("logprob carrier is choice-bearing");
    debug_assert!(
        choice.logprobs.is_none(),
        "one scheduler token must attach logprobs exactly once"
    );
    choice.logprobs = Some(ChoiceLogprobs {
        content: vec![sampled],
    });
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::chat_stream::continuous_usage::ContinuousUsageTracker;
    use crate::api::chat_stream::dsv41::delta_events;
    use crate::api::dsv41::Delta;
    use crate::tool_parser::{FunctionCall, ToolCall};

    fn sampled(token: &str) -> TokenLogprobInfo {
        TokenLogprobInfo {
            token: token.to_string(),
            logprob: -0.25,
            bytes: None,
            top_logprobs: vec![TopLogprob {
                token: "alternative".to_string(),
                logprob: -1.5,
                bytes: None,
            }],
        }
    }

    fn entry_count(chunks: &[ChatCompletionChunk]) -> usize {
        chunks
            .iter()
            .flat_map(|chunk| &chunk.choices)
            .filter_map(|choice| choice.logprobs.as_ref())
            .map(|logprobs| logprobs.content.len())
            .sum()
    }

    fn carrier_count(chunks: &[ChatCompletionChunk]) -> usize {
        chunks
            .iter()
            .flat_map(|chunk| &chunk.choices)
            .filter(|choice| choice.logprobs.is_some())
            .count()
    }

    #[test]
    fn buffered_or_suppressed_token_gets_one_logprob_progress_frame() {
        let chunks = attach_choice_logprobs(Vec::new(), "m", "id", sampled("hidden"));
        let mut usage = ContinuousUsageTracker::new(true, true, 7, "m", "id");
        let chunks = usage.observe_token(chunks);

        assert_eq!(chunks.len(), 1);
        assert_eq!(carrier_count(&chunks), 1);
        assert_eq!(entry_count(&chunks), 1);
        assert_eq!(chunks[0].choices[0].delta.content, None);
        assert_eq!(
            chunks[0]
                .usage
                .as_ref()
                .map(|usage| usage.completion_tokens),
            Some(1),
        );
    }

    #[test]
    fn dsv41_multiframe_token_attaches_once_and_counts_usage_once() {
        let call = ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "lookup".to_string(),
                arguments: r#"{"q":"x"}"#.to_string(),
            },
        };
        let mut chunks = Vec::new();
        delta_events("m", "id", Delta::ToolCall { index: 0, call }, &mut chunks);
        assert_eq!(chunks.len(), 2);

        let chunks = attach_choice_logprobs(chunks, "m", "id", sampled("tool-token"));
        let mut usage = ContinuousUsageTracker::new(true, true, 7, "m", "id");
        let chunks = usage.observe_token(chunks);

        assert_eq!(chunks.len(), 2);
        assert_eq!(carrier_count(&chunks), 1);
        assert_eq!(entry_count(&chunks), 1);
        assert!(chunks.iter().all(|chunk| {
            chunk
                .usage
                .as_ref()
                .is_some_and(|usage| usage.completion_tokens == 1)
        }));
    }

    #[test]
    fn generic_visible_token_preserves_logprob_payload_cardinality() {
        let chunks = vec![ChatCompletionChunk::content_chunk(
            "m",
            "id",
            "visible".to_string(),
        )];
        let chunks = attach_choice_logprobs(chunks, "m", "id", sampled("visible"));

        assert_eq!(chunks.len(), 1);
        assert_eq!(carrier_count(&chunks), 1);
        assert_eq!(entry_count(&chunks), 1);
        let content = &chunks[0].choices[0]
            .logprobs
            .as_ref()
            .expect("logprobs attached")
            .content;
        assert_eq!(content[0].token, "visible");
        assert_eq!(content[0].top_logprobs.len(), 1);
    }
}
