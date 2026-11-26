// SPDX-License-Identifier: AGPL-3.0-only
//
// Next-token bitmask tests — adapted from
// `test_grammar_matcher_basic.py` (`test_token_operations`,
// `test_fill_next_token_bitmask`) and `test_token_bitmask_operations.py`.
//
// Cover: every accepted token's bit set / rejected clear, the
// uncertain-token resolution path (multi-char tokens against a
// char-class grammar), JSON-schema + structural-tag compiled grammars,
// the all-accepting return value, and the `FillError` guards.

use crate::compiler::GrammarCompiler;

use super::super::bitmask::bitmask_size;
use super::super::{BatchGrammarMatcher, FillError, TokenBitmask};
use super::{STOP_ID, accepted_ids, compile_ebnf, id_of, matcher, tok, wrap};

// ----- TokenBitmask packed-layout unit tests ------------------------

#[test]
fn token_bitmask_set_get_roundtrip() {
    let mut b = TokenBitmask::new(70);
    assert_eq!(b.count_set(), 0);
    b.set(0, true);
    b.set(31, true);
    b.set(32, true);
    b.set(69, true);
    assert!(b.is_set(0) && b.is_set(31) && b.is_set(32) && b.is_set(69));
    assert!(!b.is_set(1) && !b.is_set(33));
    assert_eq!(b.count_set(), 4);
}

#[test]
fn token_bitmask_fill_all_respects_vocab_padding() {
    // vocab 33 -> 2 words, but only 33 logical bits.
    let mut b = TokenBitmask::new(33);
    b.fill_all();
    assert!(b.all_set());
    assert_eq!(b.count_set(), 33);
    b.clear();
    assert_eq!(b.count_set(), 0);
}

#[test]
fn token_bitmask_rejected_tokens_list() {
    let mut b = TokenBitmask::new(10);
    b.fill_all();
    b.set(3, false);
    b.set(7, false);
    assert_eq!(b.rejected_tokens(), vec![3, 7]);
}

// ----- fill_next_token_bitmask: accepted set correctness ------------

#[test]
fn fill_initial_accepts_only_legal_first_tokens() {
    // root ::= "abc" — legal first tokens are exactly those that are a
    // prefix of "abc": "a", "ab", "abc".
    let mut m = matcher("root ::= \"abc\"\n");
    let acc = accepted_ids(&mut m);
    assert!(acc.contains(&id_of("a")));
    assert!(acc.contains(&id_of("ab")));
    assert!(acc.contains(&id_of("abc")));
    // "b", "x", and the stop token are illegal here.
    assert!(!acc.contains(&id_of("b")));
    assert!(!acc.contains(&id_of("x")));
    assert!(!acc.contains(&STOP_ID));
}

#[test]
fn fill_includes_stop_token_when_root_complete() {
    let mut m = matcher("root ::= \"a\"\n");
    assert!(m.accept_token(id_of("a"), false));
    // Root complete: the stop token is now accepted.
    let acc = accepted_ids(&mut m);
    assert!(acc.contains(&STOP_ID));
}

#[test]
fn fill_excludes_stop_token_when_root_incomplete() {
    let mut m = matcher("root ::= \"abc\"\n");
    assert!(m.accept_token(id_of("a"), false));
    let acc = accepted_ids(&mut m);
    assert!(!acc.contains(&STOP_ID));
}

#[test]
fn fill_never_accepts_special_tokens() {
    // Token id 18 is the empty-decode special token.
    let mut m = matcher("root ::= [a-d]+\n");
    let acc = accepted_ids(&mut m);
    assert!(!acc.contains(&18));
}

#[test]
fn fill_uncertain_token_resolution_charclass() {
    // [a-d]+ : multi-char tokens "ab","abc","bc","cd","abcd" are all
    // uncertain (every char in range) and must resolve to ACCEPTED.
    let mut m = matcher("root ::= [a-d]+\n");
    let acc = accepted_ids(&mut m);
    for t in ["a", "b", "c", "d", "ab", "abc", "bc", "cd", "abcd"] {
        assert!(acc.contains(&id_of(t)), "token {t} should be accepted");
    }
    // "x" is outside [a-d].
    assert!(!acc.contains(&id_of("x")));
}

#[test]
fn fill_uncertain_token_partial_match_rejected() {
    // Grammar accepts only "ab"; the token "abc" extends past the
    // grammar end -> uncertain token must resolve to REJECTED, while
    // "ab" itself is accepted.
    let mut m = matcher("root ::= \"ab\"\n");
    let acc = accepted_ids(&mut m);
    assert!(acc.contains(&id_of("a")));
    assert!(acc.contains(&id_of("ab")));
    assert!(!acc.contains(&id_of("abc")));
    assert!(!acc.contains(&id_of("abcd")));
}

#[test]
fn fill_step_by_step_json_object() {
    // A small JSON-ish object grammar exercised token by token.
    let mut m = matcher("root ::= \"{\" \"\\\"\" \"a\" \"\\\"\" \":\" \"1\" \"}\"\n");
    let steps = ["{", "\"", "a", "\"", ":", "1", "}"];
    for s in steps {
        let acc = accepted_ids(&mut m);
        assert!(acc.contains(&id_of(s)), "step {s}: not in accepted set");
        assert!(m.accept_token(id_of(s), false));
    }
    assert!(m.is_grammar_completed());
}

#[test]
fn fill_return_value_signals_non_universal_mask() {
    // A constrained grammar -> mask is not all-accepting.
    let mut m = matcher("root ::= \"abc\"\n");
    let vocab = m.tokenizer_info().vocab_size();
    let mut buf = vec![0i32; bitmask_size(vocab)];
    let need_apply = m.fill_next_token_bitmask(&mut buf, 0, false).unwrap();
    assert!(need_apply, "constrained grammar must need the mask applied");
}

// ----- fill error guards -------------------------------------------

#[test]
fn fill_after_termination_errors() {
    let mut m = matcher("root ::= \"a\"\n");
    assert!(m.accept_token(id_of("a"), false));
    assert!(m.accept_token(STOP_ID, false));
    let vocab = m.tokenizer_info().vocab_size();
    let mut buf = vec![0i32; bitmask_size(vocab)];
    assert_eq!(
        m.fill_next_token_bitmask(&mut buf, 0, false),
        Err(FillError::Terminated)
    );
}

#[test]
fn fill_buffer_too_small_errors() {
    let mut m = matcher("root ::= \"a\"\n");
    let mut buf = vec![0i32; 0];
    assert_eq!(
        m.fill_next_token_bitmask(&mut buf, 0, false),
        Err(FillError::BufferTooSmall)
    );
}

#[test]
fn fill_indexed_slice_into_batch_buffer() {
    // A 2-matcher-wide buffer; fill matcher into slot index 1.
    let mut m = matcher("root ::= \"abc\"\n");
    let vocab = m.tokenizer_info().vocab_size();
    let words = bitmask_size(vocab);
    let mut buf = vec![0i32; words * 2];
    m.fill_next_token_bitmask(&mut buf, 1, false).unwrap();
    // Slot 0 untouched (all zero), slot 1 has the "a" bit set.
    let slot0 = wrap(&buf[0..words], vocab);
    let slot1 = wrap(&buf[words..2 * words], vocab);
    assert_eq!(slot0.count_set(), 0);
    assert!(slot1.is_set(id_of("a") as usize));
}

// ----- JSON-schema + structural-tag compiled grammars --------------

#[test]
fn fill_json_schema_grammar() {
    let c = GrammarCompiler::new(tok(), 1, true, -1);
    let schema = r#"{"type":"object","properties":{"a":{"type":"integer"}}}"#;
    let cg = c
        .compile_json_schema(schema, true, None, None, true, None)
        .expect("schema compiles");
    let mut m = super::super::GrammarMatcher::from_compiled_grammar(cg);
    // A JSON object must start with '{'.
    let acc = accepted_ids(&mut m);
    assert!(acc.contains(&id_of("{")));
    assert!(!acc.contains(&id_of("a")));
}

#[test]
fn fill_builtin_json_grammar() {
    let c = GrammarCompiler::new(tok(), 1, true, -1);
    let cg = c.compile_builtin_json_grammar().expect("builtin json");
    let mut m = super::super::GrammarMatcher::from_compiled_grammar(cg);
    let acc = accepted_ids(&mut m);
    // Builtin JSON accepts an object or array opener, plus scalars.
    assert!(!acc.is_empty());
}

// ----- batched fill -------------------------------------------------

#[test]
fn batch_fill_matches_sequential() {
    let cg = compile_ebnf("root ::= \"abc\"\n");
    let vocab = cg.tokenizer_info().vocab_size();
    let words = bitmask_size(vocab);

    let mut ms: Vec<_> = (0..4)
        .map(|_| super::super::GrammarMatcher::from_compiled_grammar(cg.clone()))
        .collect();
    let mut batch_buf = vec![0i32; words * 4];
    let bgm = BatchGrammarMatcher::new(Some(4));
    let res = bgm.fill_next_token_bitmask(&mut ms, &mut batch_buf, false);
    assert!(res.iter().all(|r| matches!(r, Ok(true))));

    // Each batch slice equals an independent single-matcher fill.
    let mut single = super::super::GrammarMatcher::from_compiled_grammar(cg);
    let mut single_buf = vec![0i32; words];
    single
        .fill_next_token_bitmask(&mut single_buf, 0, false)
        .unwrap();
    for i in 0..4 {
        assert_eq!(&batch_buf[i * words..(i + 1) * words], &single_buf[..]);
    }
}

#[test]
fn batch_fill_sequential_path_single_thread() {
    let cg = compile_ebnf("root ::= [a-d]+\n");
    let vocab = cg.tokenizer_info().vocab_size();
    let words = bitmask_size(vocab);
    let mut ms = vec![
        super::super::GrammarMatcher::from_compiled_grammar(cg.clone()),
        super::super::GrammarMatcher::from_compiled_grammar(cg),
    ];
    let mut buf = vec![0i32; words * 2];
    let bgm = BatchGrammarMatcher::new(Some(1));
    let res = bgm.fill_next_token_bitmask(&mut ms, &mut buf, false);
    assert_eq!(res.len(), 2);
    assert!(res.iter().all(|r| r.is_ok()));
}

#[test]
fn batch_fill_empty_matchers_is_empty() {
    let bgm = BatchGrammarMatcher::new(Some(2));
    let mut ms: Vec<super::super::GrammarMatcher> = Vec::new();
    let mut buf: Vec<i32> = Vec::new();
    assert!(
        bgm.fill_next_token_bitmask(&mut ms, &mut buf, false)
            .is_empty()
    );
}

#[test]
fn bitmask_slice_truncates_oversized_buffer() {
    use super::super::BitmaskSlice;
    let vocab = 10; // needs 1 word
    let sentinel = 0x1234_5678_i32;
    let mut buf = vec![sentinel; 3];
    {
        let mut view = BitmaskSlice::new(&mut buf, vocab).expect("slice");
        assert_eq!(view.words().len(), 1);
        view.fill_all();
        view.clear();
    }
    assert_eq!(buf[0], 0);
    assert_eq!(buf[1], sentinel, "trailing word 1 must not be overwritten");
    assert_eq!(buf[2], sentinel, "trailing word 2 must not be overwritten");
}

#[test]
fn override_stop_token_enabled_in_rejected_store_type_branch() {
    // `[^z]*` accepts almost every token in `tok()` -> StoreType::Rejected,
    // and root can complete immediately. Override stop token to id 18 (which
    // is a special token) and confirm it is still set in the bitmask.
    let cg = compile_ebnf("root ::= [^z]*\n");
    let mut m = super::super::GrammarMatcher::new(cg, Some(vec![18]), false, -1);
    let acc = accepted_ids(&mut m);
    assert!(
        acc.contains(&18),
        "override stop token 18 must be enabled even in StoreType::Rejected branch"
    );
}

#[test]
fn batch_fill_zero_vocab_does_not_panic() {
    use crate::tokenizer::{TokenizerInfo, VocabType};
    let empty_tok = TokenizerInfo::new(&[], VocabType::Raw, None, Some(vec![]), false);
    let c = GrammarCompiler::new(empty_tok, 1, false, -1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"a\"\n", "root")
        .unwrap();
    let mut ms = vec![
        super::super::GrammarMatcher::from_compiled_grammar(cg.clone()),
        super::super::GrammarMatcher::from_compiled_grammar(cg),
    ];
    let bgm = BatchGrammarMatcher::new(Some(2));
    let mut buf: Vec<i32> = Vec::new();
    let res = bgm.fill_next_token_bitmask(&mut ms, &mut buf, false);
    assert_eq!(res.len(), 2);
    assert!(res.iter().all(|r| r.is_ok()));
}
