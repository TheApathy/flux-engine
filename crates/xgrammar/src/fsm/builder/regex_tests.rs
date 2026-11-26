// SPDX-License-Identifier: AGPL-3.0-only
//
// Unit tests for the sibling module, in a separate file so the
// code file stays under the 250-line cap (included via `#[path]`).

use super::*;

fn build(re: &str) -> FsmWithStartEnd {
    build_regex(re).unwrap_or_else(|e| panic!("build {re}: {e}"))
}

#[test]
fn literal_with_escape() {
    let f = build("abcd\\n");
    assert!(f.accept_string(b"abcd\n"));
    assert!(!f.accept_string(b"abcd"));
}

#[test]
fn unicode_literal_byte_wise() {
    let f = build("你好a");
    assert!(f.accept_string("你好a".as_bytes()));
}

#[test]
fn empty_groups_accept_empty() {
    let f = build("(())()()");
    assert!(f.accept_string(b""));
}

#[test]
fn alternation() {
    let f = build("aaa|[\\d]");
    assert!(f.accept_string(b"aaa"));
    assert!(f.accept_string(b"1"));
    assert!(!f.accept_string(b"aa"));
}

#[test]
fn nested_groups_and_alternation() {
    let f = build("(([\\d]|[\\w])|aaa)");
    assert!(f.accept_string(b"aaa"));
    assert!(f.accept_string(b"1"));
    assert!(!f.accept_string(b"1a"));
}

#[test]
fn plus_quantifier() {
    let f = build("1[\\d]+");
    assert!(f.accept_string(b"1111"));
    assert!(!f.accept_string(b"1"));
}

#[test]
fn star_quantifier() {
    let f = build("1[1]*");
    assert!(f.accept_string(b"1"));
    assert!(f.accept_string(b"1111"));
}

#[test]
fn optional_quantifier() {
    let f = build("1[\\d]?");
    assert!(f.accept_string(b"1"));
    assert!(f.accept_string(b"11"));
    assert!(!f.accept_string(b"1111"));
}

#[test]
fn quantifiers_on_space() {
    let f = build(" * * + ? *");
    assert!(f.accept_string(b" "));
    assert!(f.accept_string(b"      "));
}

#[test]
fn connection_with_classes() {
    let f = build(" [a-zA-Z0-9]--");
    assert!(f.accept_string(b" a--"));
}

#[test]
fn repeat_bounded() {
    let f = build("[\\d]{1,5}");
    assert!(f.accept_string(b"123"));
    assert!(f.accept_string(b"12345"));
    assert!(!f.accept_string(b"123456"));
}

#[test]
fn repeat_exact() {
    let f = build("[\\d]{6}");
    assert!(f.accept_string(b"123456"));
    assert!(!f.accept_string(b"1234567"));
}

#[test]
fn repeat_open_ended() {
    let f = build("[\\d]{6, }");
    assert!(f.accept_string(b"123456"));
    assert!(f.accept_string(b"1234567"));
    assert!(!f.accept_string(b"12345"));
}

#[test]
fn integrated_pattern() {
    let f = build("((naive|bbb|[\\d]+)*[\\w])|  +");
    for s in [&b"naive1"[..], b"bbbnaive114514W", b"    ", b"123", b"_"] {
        assert!(f.accept_string(s), "should accept {s:?}");
    }
    for s in [&b"naive"[..], b"naive   ", b"123 ", b"aaa"] {
        assert!(!f.accept_string(s), "should reject {s:?}");
    }
}

#[test]
fn email_like_pattern() {
    let f = build(r"(\w+)(\.\w+)*@(\w+)(\.\w+)+");
    assert!(f.accept_string(b"ajidoa@a.test"));
    assert!(f.accept_string(b"as____________as@abc.me.test"));
    assert!(!f.accept_string(b"@google.test"));
    assert!(!f.accept_string(b"hello@"));
    assert!(!f.accept_string(b"hello"));
}

#[test]
fn time_like_pattern() {
    let f = build(r"(\d{1,2}):(\d{2})(:(\d{2}))?");
    for s in [&b"1:34"[..], b"23:59", b"00:00", b"01:02:03"] {
        assert!(f.accept_string(s), "accept {s:?}");
    }
    for s in [&b"19"[..], b"12:6", b"12:34:", b"::"] {
        assert!(!f.accept_string(s), "reject {s:?}");
    }
}

#[test]
fn invalid_regex_errors() {
    assert!(build_regex("+abc").is_err());
    assert!(build_regex("abc)").is_err());
    assert!(build_regex("[[a]").is_err());
    assert!(build_regex("abc\\").is_err());
    assert!(build_regex("[abc").is_err());
    assert!(build_regex("[abc\\").is_err());
    assert!(build_regex("a{5,2}").is_err());
}

#[test]
fn escaped_range_uses_actual_lower_bound() {
    // [0-\A] must span '0' (0x30) ..= 'A' (0x41), not '[' (0x5B) ..= 'A'.
    let f = build(r"[0-\A]");
    assert!(f.accept_string(b"0"));
    assert!(f.accept_string(b"9"));
    assert!(f.accept_string(b"A"));
    assert!(!f.accept_string(b"["));
    assert!(!f.accept_string(b"B"));
}

#[test]
fn repeat_zero_and_open_lower_bounds() {
    let f0 = build("a{0}");
    assert!(f0.accept_string(b""));
    assert!(!f0.accept_string(b"a"));

    let f02 = build("a{0,2}");
    assert!(f02.accept_string(b""));
    assert!(f02.accept_string(b"a"));
    assert!(f02.accept_string(b"aa"));
    assert!(!f02.accept_string(b"aaa"));

    let f1inf = build("a{1,}");
    assert!(!f1inf.accept_string(b""));
    assert!(f1inf.accept_string(b"a"));
    assert!(f1inf.accept_string(b"aa"));

    let f0inf = build("a{0,}");
    assert!(f0inf.accept_string(b""));
    assert!(f0inf.accept_string(b"a"));
    assert!(f0inf.accept_string(b"aaa"));
}
