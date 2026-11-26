// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::grammar::functor::normalizer::GrammarNormalizer;
use crate::grammar::parse_ebnf_default;

fn normed(ebnf: &str) -> GrammarData {
    GrammarNormalizer::apply(parse_ebnf_default(ebnf).expect("parse"))
}

#[test]
fn used_rules_skips_unreferenced() {
    let g = normed("root ::= \"a\"\norphan ::= \"b\"\n");
    let used = UsedRulesAnalyzer::apply(&g);
    assert_eq!(used, vec![g.root_rule_id()]);
}

#[test]
fn used_rules_follows_refs() {
    let g = normed("root ::= sub\nsub ::= \"x\"\n");
    let used = UsedRulesAnalyzer::apply(&g);
    assert_eq!(used.len(), 2);
}

#[test]
fn ref_graph_is_inverted() {
    let g = normed("root ::= sub\nsub ::= \"x\"\n");
    let graph = RuleRefGraphFinder::apply(&g);
    let sub_id = (0..g.num_rules())
        .find(|&i| g.rule(i).name == "sub")
        .unwrap();
    // sub is referenced by root
    assert!(graph[sub_id as usize].contains(&g.root_rule_id()));
}

#[test]
fn allow_empty_detects_explicit() {
    let g = normed("root ::= \"\" | \"a\"\n");
    let empty = AllowEmptyRuleAnalyzer::apply(&g);
    assert!(empty.contains(&g.root_rule_id()));
}

#[test]
fn allow_empty_detects_indirect() {
    let g = normed("root ::= sub\nsub ::= \"\"\n");
    let empty = AllowEmptyRuleAnalyzer::apply(&g);
    assert_eq!(empty.len(), 2);
}

#[test]
fn non_empty_rule_excluded() {
    let g = normed("root ::= \"abc\"\n");
    let empty = AllowEmptyRuleAnalyzer::apply(&g);
    assert!(empty.is_empty());
}

#[test]
fn allow_empty_handles_stop_str_tag_dispatch_with_nullable_target() {
    // stop_eos=false TagDispatch targeting a nullable rule must not panic
    // and must not mark the TagDispatch rule itself as nullable.
    let g = normed(
        "root ::= TagDispatch((\"t\", sub), stop_eos=false, stop_str=(\"END\"))\nsub ::= \"\" | \"x\"\n",
    );
    let empty = AllowEmptyRuleAnalyzer::apply(&g);
    let root_id = g.root_rule_id();
    let sub_id = (0..g.num_rules())
        .find(|&i| g.rule(i).name == "sub")
        .unwrap();
    assert!(empty.contains(&sub_id));
    assert!(!empty.contains(&root_id));
}

#[test]
fn allow_empty_seeds_zero_min_repeat_over_non_nullable_rule() {
    use crate::grammar::builder::GrammarBuilder;
    let mut b = GrammarBuilder::new();
    let sub_body_bs = b.add_byte_string("a");
    let sub_seq = b.add_sequence(&[sub_body_bs]);
    let sub_choices = b.add_choices(&[sub_seq]);
    let sub_id = b.add_rule_named("sub", sub_choices).unwrap();
    let rep = b.add_repeat(sub_id, 0, 5);
    let root_seq = b.add_sequence(&[rep]);
    let root_choices = b.add_choices(&[root_seq]);
    let root_id = b.add_rule_named("root", root_choices).unwrap();
    let g = b.get("root").unwrap();
    let empty = AllowEmptyRuleAnalyzer::apply(&g);
    assert!(empty.contains(&root_id));
    assert!(!empty.contains(&sub_id));
}
