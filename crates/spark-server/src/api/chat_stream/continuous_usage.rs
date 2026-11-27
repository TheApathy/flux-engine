// SPDX-License-Identifier: AGPL-3.0-only

use crate::openai::{ChatCompletionChunk, Usage};

/// Request-scoped cumulative usage state for one chat-completion stream.
///
/// The tracker counts committed scheduler events, stamps the same cumulative
/// count on every wire chunk produced by one token, and emits a normal
/// empty-delta choice when a committed token has no visible text yet. Keeping
/// the progress frame choice-bearing avoids overloading OpenAI's terminal
/// `choices: []` usage frame.
pub(super) struct ContinuousUsageTracker {
    enabled: bool,
    prompt_tokens: usize,
    completion_tokens: usize,
    model: String,
    id: String,
}

impl ContinuousUsageTracker {
    pub(super) fn new(
        include_usage: bool,
        continuous_usage_stats: bool,
        prompt_tokens: usize,
        model: &str,
        id: &str,
    ) -> Self {
        Self {
            enabled: include_usage && continuous_usage_stats,
            prompt_tokens,
            completion_tokens: 0,
            model: model.to_string(),
            id: id.to_string(),
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(super) fn decorate_role(&self, chunk: ChatCompletionChunk) -> ChatCompletionChunk {
        if self.enabled {
            chunk.with_usage(self.live_usage())
        } else {
            chunk
        }
    }

    pub(super) fn observe_token(
        &mut self,
        chunks: Vec<ChatCompletionChunk>,
    ) -> Vec<ChatCompletionChunk> {
        self.observe_token_batch(1, chunks)
    }

    /// Advance cumulative usage by `token_count` committed tokens from a
    /// coalesced `StreamEvent::TokenBatch`, stamping the updated count on every
    /// choice chunk (or synthesizing one progress chunk when empty).
    pub(super) fn observe_token_batch(
        &mut self,
        token_count: usize,
        mut chunks: Vec<ChatCompletionChunk>,
    ) -> Vec<ChatCompletionChunk> {
        if !self.enabled || token_count == 0 {
            return chunks;
        }
        self.completion_tokens = self.completion_tokens.saturating_add(token_count);
        if chunks.is_empty() {
            chunks.push(ChatCompletionChunk::progress_chunk(&self.model, &self.id));
        }
        self.stamp_choice_chunks(&mut chunks);
        chunks
    }

    pub(super) fn observe_completion_progress(&mut self) -> Vec<ChatCompletionChunk> {
        self.observe_token(Vec::new())
    }

    /// Reconcile the live counter with the scheduler's authoritative terminal
    /// count before any buffered output or finish frame reaches the client.
    pub(super) fn reconcile_done(
        &mut self,
        completion_tokens: usize,
        mut chunks: Vec<ChatCompletionChunk>,
    ) -> Vec<ChatCompletionChunk> {
        if !self.enabled {
            return chunks;
        }
        if completion_tokens != self.completion_tokens {
            tracing::warn!(
                observed_completion_tokens = self.completion_tokens,
                scheduler_completion_tokens = completion_tokens,
                "continuous usage reconciled at stream completion"
            );
        }
        if completion_tokens < self.completion_tokens {
            // This should be unreachable: streamed/progress events represent
            // committed tokens and scheduler Done must include every one.
            tracing::error!(
                observed_completion_tokens = self.completion_tokens,
                scheduler_completion_tokens = completion_tokens,
                "scheduler completion count regressed below streamed usage"
            );
        }
        self.completion_tokens = completion_tokens;
        self.stamp_choice_chunks(&mut chunks);
        // Preserve terminal timing/detail fields while making the final
        // choices-empty usage frame agree with the monotonic live count.
        for chunk in &mut chunks {
            if chunk.choices.is_empty()
                && let Some(usage) = chunk.usage.as_mut()
            {
                usage.completion_tokens = self.completion_tokens;
                usage.total_tokens = usage.prompt_tokens + self.completion_tokens;
            }
        }
        chunks
    }

    fn stamp_choice_chunks(&self, chunks: &mut [ChatCompletionChunk]) {
        let usage = self.live_usage();
        for chunk in chunks {
            if !chunk.choices.is_empty() {
                chunk.usage = Some(usage.clone());
            }
        }
    }

    fn live_usage(&self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.prompt_tokens + self.completion_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: None,
            time_to_first_token_ms: 0.0,
            response_tokens_per_second: 0.0,
        }
    }
}

/// OpenAI/vLLM terminal order: finish choice, optional final usage, `[DONE]`.
pub(super) fn terminal_chunks(
    model: &str,
    id: &str,
    finish_reason: &str,
    usage: Usage,
    include_usage: bool,
) -> Vec<ChatCompletionChunk> {
    if include_usage {
        vec![
            ChatCompletionChunk::final_chunk_no_usage(model, id, finish_reason),
            ChatCompletionChunk::usage_only_chunk(model, id, usage),
        ]
    } else {
        vec![ChatCompletionChunk::done_chunk(
            model,
            id,
            finish_reason,
            usage,
        )]
    }
}
