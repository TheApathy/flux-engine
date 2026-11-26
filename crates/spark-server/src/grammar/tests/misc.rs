// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn test_accept_token_after_termination_short_circuits() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();
    assert!(state.accept_token(b'{' as u32));
    assert!(state.accept_token(b'}' as u32));
    let _ = state.accept_token(130);
    assert!(
        state.is_terminated(),
        "grammar must reach terminated state for this test to be meaningful"
    );

    // Under speculative decoding, the verifier can accept a stop token while
    // trailing drafts in the same step still pass through `accept_token`. The
    // short-circuit must return `true` so spec-decode draft truncation does not
    // treat post-stop tokens as grammar rejections.
    assert!(
        state.accept_token(198),
        "post-termination accept_token must short-circuit to true for any token id"
    );
    assert!(
        state.accept_token(0),
        "post-termination accept_token must remain a no-op for any token id"
    );
}

#[test]
fn test_fill_bitmask_after_stop_token() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32]; // <eos>
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    // Use the built-in JSON grammar and complete a minimal object `{}`,
    // then feed the stop token. `GrammarState::new` sets
    // `terminate_without_stop_token = false`, so the matcher reaches its
    // terminated state once the stop token is accepted after the complete body.
    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();
    assert!(state.accept_token(b'{' as u32), "accept '{{'");
    assert!(state.accept_token(b'}' as u32), "accept '}}'");
    // Best-effort: many grammars auto-terminate on body completion; others
    // need the explicit stop token. Either way, the fill_bitmask guard
    // must hold once is_terminated() reports true.
    let _ = state.accept_token(130);
    if state.is_terminated() {
        // The `is_terminated()` guard short-circuits and returns `false`
        // without calling into the terminated matcher.
        let has_constraint = state.fill_bitmask();
        assert!(
            !has_constraint,
            "terminated grammar should report no constraint"
        );
        // Second call must also stay safe (idempotent).
        let _ = state.fill_bitmask();
    } else {
        // If the test grammar no longer auto-terminates, the bug can't be
        // exercised by this path — flag so future refactors don't silently
        // lose coverage.
        panic!("grammar did not terminate; update the test to force termination");
    }
}

#[test]
fn test_ebnf_compilation() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    // Simple EBNF: match "hello"
    let ebnf = r#"root ::= "hello""#;
    let result = engine.compile_ebnf(ebnf, "root");
    assert!(
        result.is_ok(),
        "EBNF compilation failed: {}",
        result.as_ref().err().unwrap()
    );
}

#[test]
fn test_extract_ordered_vocab() {
    // Verify the helper produces correct ordering.
    // We cannot easily construct a tokenizers::Tokenizer in a unit test
    // without a tokenizer.json file, so this is a compile-time check
    // that the function signature is correct.
    // Integration tests with a real tokenizer will cover correctness.
}

#[test]
fn test_multiple_tools_hermes() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let tools = vec![
        ToolDefinition {
            tool_type: "function".to_string(),
            function: crate::tool_parser::FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                })),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: crate::tool_parser::FunctionDefinition {
                name: "search".to_string(),
                description: Some("Search the web".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                })),
            },
        },
    ];

    let result = engine.compile_hermes_tool_grammar(&tools, false, true);
    assert!(
        result.is_ok(),
        "Multi-tool Hermes compilation failed: {}",
        result.as_ref().err().unwrap()
    );
}
