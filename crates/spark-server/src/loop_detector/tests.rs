// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn sig_text_only(text: &str) -> Signature {
    Signature::build(text, std::iter::empty())
}

fn sig_with_tool(text: &str, name: &str, args: &str) -> Signature {
    Signature::build(text, std::iter::once((name, args)))
}

#[test]
fn empty_recent_returns_none() {
    assert_eq!(detect(&[]), LoopState::None);
    assert_eq!(
        detect(&[sig_text_only("hello world how are you doing today")]),
        LoopState::None
    );
}

#[test]
fn three_distinct_messages_no_loop() {
    let recent = vec![
        sig_text_only("the quick brown fox jumps over the lazy dog"),
        sig_text_only("once upon a time there was a small village"),
        sig_text_only("rust is a systems programming language designed for safety"),
    ];
    assert_eq!(detect(&recent), LoopState::None);
}

#[test]
fn three_identical_intros_fire_loop() {
    // Cross-turn prose loop — failure pattern from dump 2026-04-25
    // seq=35 msgs[11/15/17/19].
    let intro = "I will create a proper echo server for you. Let me write \
                     a clean implementation with axum and serde.";
    let recent = vec![
        sig_text_only(intro),
        sig_text_only(intro),
        sig_text_only(intro),
    ];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Suppress { .. } | LoopState::Hint { .. }),
        "got {v:?}"
    );
}

#[test]
fn identical_tool_args_fire_loop() {
    // Repeated identical `cargo init` tool invocation loop.
    let cmd = r#"{"command":"mkdir -p /tmp/axum-test-5 && cd /tmp/axum-test-5 && cargo init --name axum_echo_server"}"#;
    let recent = vec![
        sig_with_tool("", "Bash", cmd),
        sig_with_tool("", "Bash", cmd),
        sig_with_tool("", "Bash", cmd),
        sig_with_tool("", "Bash", cmd),
    ];
    let v = detect(&recent);
    assert!(matches!(v, LoopState::Suppress { .. }), "got {v:?}");
}

#[test]
fn different_tool_args_with_same_name_do_not_fire() {
    let recent = vec![
        sig_with_tool("", "Bash", r#"{"command":"ls /a"}"#),
        sig_with_tool("", "Bash", r#"{"command":"echo hi"}"#),
        sig_with_tool("", "Bash", r#"{"command":"pwd"}"#),
    ];
    // Token overlap is "Bash command" — high enough to fire on
    // small sets. Our threshold should resist this; if it
    // fails, raise SHINGLE_ORDER.
    let v = detect(&recent);
    assert_eq!(v, LoopState::None, "got {v:?}");
}

#[test]
fn slightly_varied_intros_still_fire() {
    // The model often paraphrases its intro slightly — must
    // catch this too.
    let a = "I'll create a proper echo server using axum and serde with full tests.";
    let b = "I'll create a proper echo server using axum and serde with passing tests.";
    let c = "I'll create a proper echo server using axum and serde, including tests.";
    let recent = vec![sig_text_only(a), sig_text_only(b), sig_text_only(c)];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Hint { .. } | LoopState::Suppress { .. }),
        "near-identical paraphrases must trigger detection: {v:?}"
    );
}

#[test]
fn moderate_similarity_3_turns_now_suppress() {
    // Regression for threshold recalibration.
    // Four turns where the assistant repeats the same core sentence with a
    // small trailing variation (Jaccard ~0.52–0.73 across pairs).
    //
    // With HIGH=0.65 and MODERATE=0.50, `Suppress` fires rather than `Hint`.
    let common = "Let me create the necessary files for the Axum server with an echo endpoint and passing tests in the project directory now";
    let recent = vec![
        sig_text_only(&format!("{common} immediately")),
        sig_text_only(&format!("{common} carefully")),
        sig_text_only(&format!("{common} step by step")),
        sig_text_only(&format!("{common} as planned")),
    ];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Suppress { .. }),
        "post-recalibration: 4 turns sharing the same long prose stem must Suppress: {v:?}"
    );
}

#[test]
fn moderate_band_with_065_max_suppress() {
    // After lowering HIGH_SIMILARITY to 0.65, a 3-turn run whose
    // max pair similarity lands inside the [0.65, 0.80) band must
    // escalate to Suppress (was Hint at the old 0.80 bar).
    // 13 distinct shared words (10 shared 4-shingles) + 2 distinct trailing
    // words per turn -> pairwise Jaccard = 10 / (10 + 4) = 0.7143.
    let s = "the quick brown fox jumps over every lazy hound near tall oak trees";
    let recent = vec![
        sig_text_only(&format!("{s} alpha one")),
        sig_text_only(&format!("{s} bravo two")),
        sig_text_only(&format!("{s} charlie three")),
    ];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Suppress { score, .. } if (0.65..0.80).contains(&score)),
        "3-turn run with Jaccard in [0.65, 0.80) must Suppress: {v:?}"
    );
}

#[test]
fn empty_signatures_are_skipped_not_counted() {
    // A 4-byte heartbeat in the middle should not break a real
    // 3-turn loop on either side of it.
    let intro =
        "I will create a proper echo server for you. Let me write a clean implementation now.";
    let recent = vec![
        sig_text_only(intro),
        sig_text_only(""), // empty — skipped by detector
        sig_text_only(intro),
        sig_text_only(intro),
    ];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Suppress { .. } | LoopState::Hint { .. }),
        "got {v:?}"
    );
}

#[test]
fn one_off_repeat_does_not_trigger_suppress() {
    // Two identical messages is normal (e.g., the model retries
    // a transient failure). Must NOT escalate to Suppress on two
    // turns alone.
    let intro =
        "I will create a proper echo server for you. Let me write a clean implementation now.";
    let recent = vec![sig_text_only(intro), sig_text_only(intro)];
    let v = detect(&recent);
    assert_eq!(v, LoopState::None, "two-turn repeat is not yet a loop");
}

#[test]
fn signature_below_min_tokens_is_empty() {
    // Even if some shingles could be formed at order < SHINGLE_ORDER,
    // we explicitly zero short channels so trivial echoes don't
    // produce noise.
    let s = Signature::build("yes", std::iter::empty());
    assert!(
        s.is_empty(),
        "1-token text below MIN_CHANNEL_TOKENS must yield empty signature"
    );
}
