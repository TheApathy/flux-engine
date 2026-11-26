// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::grammar::functor::normalizer::GrammarNormalizer;
use crate::grammar::parse_ebnf_default;
use crate::grammar::printer::print_grammar;

fn normed(ebnf: &str) -> GrammarData {
    GrammarNormalizer::apply(parse_ebnf_default(ebnf).expect("parse"))
}

#[test]
fn byte_string_fuser_merges() {
    let g = normed("root ::= \"a\" \"b\" \"c\"\n");
    let fused = ByteStringFuser::apply(g);
    let out = print_grammar(&fused);
    assert!(out.contains("\"abc\""), "{out}");
}

#[test]
fn dead_code_eliminator_drops_unused() {
    let g = normed("root ::= \"a\"\nunused ::= \"b\"\n");
    let pruned = DeadCodeEliminator::apply(g);
    assert_eq!(pruned.num_rules(), 1);
}

#[test]
fn dead_code_keeps_referenced() {
    let g = normed("root ::= sub\nsub ::= \"x\"\n");
    let pruned = DeadCodeEliminator::apply(g);
    assert_eq!(pruned.num_rules(), 2);
}

#[test]
fn rule_inliner_inlines_leading_ref() {
    let g = normed("root ::= sub \"z\"\nsub ::= \"a\" | \"b\"\n");
    let inlined = RuleInliner::apply(g);
    let out = print_grammar(&inlined);
    // After inlining, root should reference "a"/"b" directly.
    assert!(out.contains('a') && out.contains('b'), "{out}");
}

#[test]
fn optimizer_sets_flag() {
    let g = normed("root ::= \"hello\"\n");
    let opt = GrammarOptimizer::apply(g);
    assert!(opt.optimized);
}

#[test]
fn optimizer_builds_fsms() {
    let g = normed("root ::= \"ab\" | \"cd\"\n");
    let opt = GrammarOptimizer::apply(g);
    assert_eq!(opt.per_rule_fsms.len(), opt.num_rules() as usize);
    assert!(opt.per_rule_fsms[0].is_some());
}

#[test]
fn non_ascii_tag_dispatch_survives_full_optimizer_pipeline() {
    let marker = "<\u{ff5c}DSML\u{ff5c}function_calls>";
    let ebnf =
        format!("root ::= TagDispatch((\"{marker}\", call), stop_eos=true)\ncall ::= \"ok\"\n");
    let g = normed(&ebnf);
    let opt = GrammarOptimizer::apply(g);
    let body_id = opt.root_rule().body_expr_id;
    let td = opt.tag_dispatch(body_id);
    assert_eq!(td.tag_rule_pairs[0].0, marker);
}

#[test]
fn mutators_preserve_is_exact_lookahead() {
    let g = normed("root ::= sub \"z\"\nsub ::= \"a\"\n");
    let analyzed = crate::grammar::functor::lookahead::LookaheadAssertionAnalyzer::apply(g);
    let sub_id = (0..analyzed.num_rules())
        .find(|&i| analyzed.rule(i).name == "sub")
        .unwrap();
    assert!(analyzed.rule(sub_id).is_exact_lookahead);

    let fused = ByteStringFuser::apply(analyzed.clone());
    assert!(fused.rule(sub_id).is_exact_lookahead);

    let pruned = DeadCodeEliminator::apply(analyzed);
    let pruned_sub = (0..pruned.num_rules())
        .find(|&i| pruned.rule(i).name == "sub")
        .unwrap();
    assert!(pruned.rule(pruned_sub).is_exact_lookahead);
}
