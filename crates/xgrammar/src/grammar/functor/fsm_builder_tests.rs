// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::grammar::functor::normalizer::GrammarNormalizer;
use crate::grammar::parse_ebnf_default;

fn built(ebnf: &str) -> GrammarData {
    let mut g = GrammarNormalizer::apply(parse_ebnf_default(ebnf).expect("parse"));
    GrammarFsmBuilder::apply(&mut g);
    g
}

#[test]
fn builds_fsm_for_byte_string() {
    let g = built("root ::= \"hello\"\n");
    let fsm = g.per_rule_fsms[g.root_rule_id() as usize].as_ref().unwrap();
    assert!(fsm.accept_string(b"hello"));
    assert!(!fsm.accept_string(b"world"));
}

#[test]
fn builds_fsm_for_choices() {
    let g = built("root ::= \"ab\" | \"cd\"\n");
    let fsm = g.per_rule_fsms[g.root_rule_id() as usize].as_ref().unwrap();
    assert!(fsm.accept_string(b"ab"));
    assert!(fsm.accept_string(b"cd"));
    assert!(!fsm.accept_string(b"ac"));
}

#[test]
fn builds_fsm_for_char_class() {
    let g = built("root ::= [a-z]\n");
    let fsm = g.per_rule_fsms[g.root_rule_id() as usize].as_ref().unwrap();
    assert!(fsm.accept_string(b"q"));
    assert!(!fsm.accept_string(b"Q"));
}

#[test]
fn builds_fsm_for_star_class() {
    let g = built("root ::= [0-9]*\n");
    let fsm = g.per_rule_fsms[g.root_rule_id() as usize].as_ref().unwrap();
    assert!(fsm.accept_string(b""));
    assert!(fsm.accept_string(b"12345"));
}

#[test]
fn negative_char_class() {
    let g = built("root ::= [^0-9]\n");
    let fsm = g.per_rule_fsms[g.root_rule_id() as usize].as_ref().unwrap();
    assert!(fsm.accept_string(b"a"));
    assert!(!fsm.accept_string(b"5"));
}

#[test]
fn negative_char_class_excludes_non_ascii_codepoints() {
    let g = built("root ::= [^\u{ff5c}]\n");
    let fsm = g.per_rule_fsms[g.root_rule_id() as usize].as_ref().unwrap();
    assert!(!fsm.accept_string("\u{ff5c}".as_bytes()));
    assert!(fsm.accept_string("\u{ff5b}".as_bytes()));
    assert!(fsm.accept_string("\u{ff5d}".as_bytes()));
    assert!(fsm.accept_string("a".as_bytes()));
    assert!(fsm.accept_string("😀".as_bytes()));
}

#[test]
fn per_rule_fsms_share_complete_fsm_allocation() {
    let g = built("root ::= a b\na ::= \"x\"\nb ::= \"y\"\n");
    for fsm in g.per_rule_fsms.iter().flatten() {
        assert!(std::sync::Arc::ptr_eq(&g.complete_fsm, &fsm.fsm));
    }
}

#[test]
fn complete_fsm_has_states() {
    let g = built("root ::= \"x\"\n");
    assert!(g.complete_fsm.num_states() > 0);
    assert_eq!(g.per_rule_fsms.len(), g.num_rules() as usize);
}

#[test]
fn empty_string_rule_fsm() {
    let g = built("root ::= \"\"\n");
    let fsm = g.per_rule_fsms[g.root_rule_id() as usize].as_ref().unwrap();
    assert!(fsm.accept_string(b""));
}
