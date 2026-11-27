// SPDX-License-Identifier: AGPL-3.0-only

//! HTTP API routes (axum handlers).
//!
//! This module was originally a single 8,979-line `api.rs`; wave 4b1
//! split it into cohesive sub-files under `api/`. Public items are
//! re-exported here so external callers (`crate::api::Foo`) keep
//! working unchanged.
//!
//! Sub-file layout:
//! - `compact`            — progressive context compaction + error helpers
//! - `chat` / `chat_stream` — OpenAI `/v1/chat/completions` (blocking + SSE)
//! - `completions`        — legacy `/v1/completions` + list/get models +
//!                          embeddings stub + cross-handler helpers
//! - `sanitizer`          — `<parameter=…>` leak suppression + bash bucketing
//! - `failures/`          — F-feature failure-recovery sub-systems
//!                          (stall, classification, circuit, duplicate)
//! - `stubs`              — batches/files/audio/images/moderations stubs
//! - `responses`,
//!   `responses_stream`,
//!   `responses_translate` — `/v1/responses` (blocking + streaming)
//! - `stored`             — stored-completion / response retrieval
//! - `conversations`      — `/v1/conversations` CRUD
//! - `misc_handlers`      — cancel / metrics / health / tokenize / detokenize
//! - `inference_types`    — `GrammarSpec`, `InferenceRequest`,
//!                          `InferenceResponse`, `StreamEvent`,
//!                          `TokenLogprobs`
//! - `inference_impl`     — `impl InferenceRequest`
//! - `strip`              — legacy `strip_thinking_tags` for completions

pub mod anthropic_count;
pub mod chat;
pub mod chat_blocking;
pub mod chat_phases;
pub mod chat_stream;
pub mod chat_stream_dispatch;
pub mod compact;
pub mod completions;
mod context_budget;
pub mod conversations;
pub mod decisions;
pub mod disagg;
pub mod dsv41;
pub mod failures;
pub mod flux_ui;
pub mod inference_impl;
pub mod inference_types;
pub mod kv_prefetch;
pub mod misc_handlers;
pub mod ollama;
pub(crate) mod request_contract;
pub mod responses;
mod responses_context_guard;
pub mod responses_stream;
pub mod responses_stream_finalize;
pub mod responses_translate;
pub mod sanitizer;
pub mod schema_check;
pub mod stored;
pub mod strip;
pub mod stubs;
pub mod telemetry;
pub mod template_inspect;

#[cfg(test)]
mod glm_tool_publication_fixture;

// Re-exports to preserve the original `crate::api::*` import surface.
// `#[allow(unused_imports)]` is applied only where the re-export is
// part of the public surface but happens to be unreferenced in this build
// (Request types kept for external clients / schema generation, plus
// `compact_messages` which is called internally from `api::chat::template`).
pub use anthropic_count::count_tokens;
pub use chat::chat_completions;
pub(crate) use chat::chat_completions_inner;
#[allow(unused_imports)]
pub use compact::compact_messages;
pub use completions::{completions, embeddings_stub, get_model, list_models};
#[allow(unused_imports)]
pub use conversations::{
    AddItemsRequest, CreateConversationRequest, UpdateConversationRequest, add_conversation_items,
    create_conversation, delete_conversation, delete_conversation_item, get_conversation,
    get_conversation_item, list_conversation_items, update_conversation,
};
#[allow(unused_imports)]
pub use decisions::{ChoiceTokenIds, decisions_endpoint, logprob_token_ids};
pub use disagg::{disagg_decode, disagg_generate, disagg_prefill, disagg_status};
pub use inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, ResponseFormatConstraint, StreamEvent,
    TokenLogprobs,
};
pub use kv_prefetch::{kv_cache_prefetch, kv_cache_prefetch_status};
#[allow(unused_imports)]
pub use misc_handlers::{
    DetokenizeRequest, cancel_response, detokenize, health, health_live, metrics_handler, tokenize,
};
pub use ollama::{ollama_show, ollama_tags};
pub use responses::responses_endpoint;
pub use stored::{
    delete_stored_response, get_stored_completion, get_stored_response, list_response_input_items,
};
pub use stubs::{
    audio_stub, batch_get_stub, batch_list_stub, batches_stub, files_stub, images_stub,
    moderations_stub,
};
pub use telemetry::{experts_usage, memory_telemetry};
pub use template_inspect::apply_template;

/// Collapses mathematically greedy sampling (`top_k == 1` with finite `temperature > 0.0`)
/// to canonical `temperature = 0.0`.
///
/// With `top_k == 1`, dividing finite logits by a positive finite temperature is strictly
/// order-preserving, and top-k truncation leaves only the unique first-wins maximum token
/// with probability `1.0` after history penalties and `logit_bias` have already been applied.
///
/// Leaving `temperature > 0.0` alongside `top_k == 1` routes requests into the stochastic
/// speculative verification path (`dflash2_selector` writes `out_max_prob = 1.0`, disabling
/// `p_cum` verify-width truncation and locking `K = 8`, and runs the multi-pass histogram
/// rejection verifier instead of 1-pass exact greedy argmax). Normalizing at the API request
/// boundary after preset penalty resolution preserves 0-ULP output parity while routing
/// `top_k == 1` onto the exact greedy fast path.
#[must_use]
pub(crate) fn canonical_greedy_temperature(temperature: f32, top_k: u32) -> f32 {
    if top_k == 1 && temperature.is_finite() && temperature > 0.0 {
        0.0
    } else {
        temperature
    }
}

#[cfg(test)]
mod canonical_greedy_tests {
    use super::canonical_greedy_temperature;

    #[test]
    fn top_k_one_with_positive_finite_temperature_collapses_to_zero() {
        for temp in [f32::MIN_POSITIVE, 0.1, 0.7, 1.0, 2.0] {
            assert_eq!(
                canonical_greedy_temperature(temp, 1),
                0.0,
                "temperature={temp} with top_k=1 must collapse to canonical greedy 0.0"
            );
        }
    }

    #[test]
    fn non_one_top_k_and_non_positive_or_non_finite_temperatures_are_preserved() {
        for top_k in [0, 2, 20, u32::MAX] {
            assert_eq!(
                canonical_greedy_temperature(0.7, top_k),
                0.7,
                "top_k={top_k} must preserve positive temperature"
            );
        }
        assert_eq!(canonical_greedy_temperature(0.0, 1), 0.0);
        assert_eq!(canonical_greedy_temperature(-0.5, 1), -0.5);
        assert!(canonical_greedy_temperature(f32::NAN, 1).is_nan());
        assert_eq!(
            canonical_greedy_temperature(f32::INFINITY, 1),
            f32::INFINITY
        );
    }
}
