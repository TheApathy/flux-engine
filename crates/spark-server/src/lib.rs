// SPDX-License-Identifier: AGPL-3.0-only

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

//! Flux Engine (`spark-server`) — shared modules for integration tests.

pub mod tokenizer;

// Python json.dumps parity; the tokenizer's chat-template `tojson` uses it.
#[path = "pyjson.rs"]
pub mod pyjson;

// The three pure modules added in PR 4 (OpenAI compat remaining items) are
// public here so `cargo test -p spark-server --lib` can exercise their
// unit tests without needing to build the full binary.
#[path = "auth.rs"]
pub mod auth;
#[path = "rate_limiter.rs"]
pub mod rate_limiter;
#[path = "refusal.rs"]
pub mod refusal;

// Load-phase debug capture and the FLUX_DEBUG file. The binary declares
// its own `mod load_debug;` / `mod debug_log;`;
// this one lets `cargo test -p spark-server --lib` run its unit tests.
#[path = "debug_log.rs"]
pub mod debug_log;
#[path = "load_debug.rs"]
pub mod load_debug;

#[path = "prefill_twophase_gate.rs"]
pub mod prefill_twophase_gate;

// GB10 unprivileged display-carveout launcher (`spark display-carveout`).
// Declared here so `cargo test -p spark-server --lib` exercises `display_carveout`,
// `rm`, and `device` unit tests without linking the full server binary.
pub mod display_carveout;

// REST retrieval draft store runtime binding. The binary crate declares its
// own `mod rest_store;` (`main.rs`) next to the scheduler call site; this
// declaration exists so `cargo test -p spark-server --lib` can exercise the
// module's unit tests without building the full binary.
#[path = "rest_store/mod.rs"]
pub mod rest_store;

// Source-level scheduler invariants that do not need CUDA symbols. Keeping
// these in the library target lets CPU-only CI prove the speculative receipt
// ordering even though the serving binary links CUDA/AOT kernels.
#[cfg(test)]
mod auxiliary_spec_source_tests;
