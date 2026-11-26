// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 (`model_type = "deepseek_v41"`) request, prompt-encoding,
//! DSML tool-parsing, grammar, and vision preprocessing layer.
//!
//! Implements the DeepSeek-V4.1 reference prompt/DSML/vision contract
//! (DeepSeek AI, MIT-licensed model checkpoint encoding scripts; see
//! `THIRD_PARTY_NOTICES.md`):
//! - [`request`]: `resolve_thinking` + `build_chat_prompt`
//! - [`encoding`]: DeepSeek-V4.1 checkpoint prompt renderer (`encoding/encoding.py`)
//! - [`parse`]: output routing, strict + tolerant DSML tool-call parsing
//! - [`grammar`]: DSML tool-call EBNF grammar builder
//!
//! Parity fixtures live under `tests/fixtures/dsv41/`.

pub(crate) mod bicubic;
pub mod encoding;
pub mod grammar;
pub mod parse;
pub(crate) use crate::pyjson;
pub mod repetition;
pub mod request;
pub mod turbojpeg;
pub mod vision;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_grammar;

use std::sync::atomic::{AtomicBool, Ordering};

// Set on every model load (a swap away turns it off again).
static SERVING: AtomicBool = AtomicBool::new(false);

/// Record whether the loaded model is deepseek_v41. The scheduler reads
/// [`serving`] to switch off the generic generation heuristics the Python
/// engine does not have (reflection suppression, think-loop watchdog, EOS
/// suppression, forced `</think>`, catastrophic-loop stop, ...).
pub fn set_serving(on: bool) {
    SERVING.store(on, Ordering::Relaxed);
}

/// Whether the loaded model is deepseek_v41 (see [`set_serving`]).
pub fn serving() -> bool {
    SERVING.load(Ordering::Relaxed)
}
