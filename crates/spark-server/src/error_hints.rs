// SPDX-License-Identifier: AGPL-3.0-only

//! What to do about an error, as opposed to what the error was.
//!
//! `{"error":{"message":"no model is loaded","type":"model_not_loaded"}}` is
//! accurate and useless: it names a state without naming the way out of it.
//! Someone who does not already know the Library exists cannot act on it.
//!
//! ## Why the hint is appended to `message` as well as emitted as `hint`
//!
//! Embedding the hint in `message` ensures every standard OpenAI/Anthropic
//! client displays the recovery action automatically without knowing about
//! Flux-specific JSON fields, while `hint` is also emitted separately for
//! clients that want to render it distinctly.

/// The actionable half of an error, keyed by the stable `error.type` string.
///
/// Keyed on `type` rather than the status code because the code is far too
/// coarse: a 503 is "model still loading" and "no model chosen" and "shutting
/// down", and only the middle one is fixed by opening the Library.
pub fn hint_for(error_type: &str) -> Option<&'static str> {
    match error_type {
        // Each hint is the ACTION only, never a restatement of the condition:
        // it is appended to a message that has just said what is wrong, and
        // "no model is loaded — No model is loaded." is how you train a reader
        // to stop reading the second half.
        "model_not_loaded" => Some(
            "open the Library (press 4 in the dashboard), choose a model and a \
             recipe, and start it; then retry this request",
        ),
        "not_ready" => Some("the socket binds before the model finishes loading, so retry shortly"),
        "shutting_down" => Some("the server is draining and will not accept new work"),
        _ => None,
    }
}

/// `message`, with the hint for `error_type` appended when there is one.
///
/// Kept separate from [`hint_for`] so the joining rule — and in particular the
/// separator, which shows up in the middle of user-visible text — has exactly
/// one definition.
pub fn message_with_hint(message: &str, error_type: &str) -> String {
    match hint_for(error_type) {
        Some(h) => format!("{message} — {h}"),
        None => message.to_string(),
    }
}

#[cfg(test)]
#[path = "error_hints_tests.rs"]
mod tests;
