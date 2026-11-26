// SPDX-License-Identifier: AGPL-3.0-only

//! Anthropic Messages API translation layer.
//!
//! Translates Anthropic `/v1/messages` requests into OpenAI
//! `ChatCompletionRequest` JSON, delegates execution to
//! `api::chat_completions_inner`, and converts the OpenAI JSON or SSE response
//! back into Anthropic format. Supports both streaming (SSE) and non-streaming.

mod block_messages;
mod handlers;
mod handlers_stream;
mod helpers;
mod image_content;
mod translate;
mod translator;
mod types;

#[cfg(test)]
mod tests;

pub use handlers::messages;
