// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarFsmBuilder — builds per-rule FSMs for an (optimized) grammar.
// Port of `GrammarFSMBuilderImpl` from `cpp/grammar_functor.cc`.
//
// Builds grammar-aware FSMs for each rule body (choices-of-sequences or
// TagDispatch), converts each to an `FsmWithStartEnd`, and splices them
// into one shared `complete_fsm`. `GrammarData::per_rule_fsms[i]` holds
// a compact view into that shared FSM for rule `i`.

use std::sync::Arc;

use crate::fsm::{CompactFsmWithStartEnd, Fsm, FsmWithStartEnd};
use crate::grammar::data::GrammarData;
use crate::grammar::expr::{GrammarExpr, GrammarExprType};

use super::char_range::{add_character_range, codepoint_to_packed_utf8};
use super::tag_dispatch_fsm::build_tag_dispatch_fsm;

/// Per-rule FSM construction pass.
pub struct GrammarFsmBuilder;

impl GrammarFsmBuilder {
    /// Build `complete_fsm` + `per_rule_fsms` for `grammar` in place.
    pub fn apply(grammar: &mut GrammarData) {
        let num_rules = grammar.num_rules();
        let mut complete = Fsm::with_states(0);
        // Each entry: `(start, ends)` plus the *sub-FSM's* node/edge
        // counts captured before splicing — needed so the per-rule
        // `CompactFsmWithStartEnd` reports the sub-FSM size rather than
        // the whole `complete_fsm` (upstream commit 58494db, #600).
        type RuleFsmSlice = (usize, Vec<bool>, usize, usize);
        let mut per_rule: Vec<Option<RuleFsmSlice>> = Vec::with_capacity(num_rules as usize);

        for i in 0..num_rules {
            let body_id = grammar.rule(i).body_expr_id;
            let body = grammar.expr(body_id);
            let rule_fsm: Option<FsmWithStartEnd> = if body.kind == GrammarExprType::TagDispatch {
                build_tag_dispatch_fsm(&grammar.tag_dispatch(body_id))
            } else {
                debug_assert_eq!(body.kind, GrammarExprType::Choices);
                build_choices(&body, grammar)
            };
            per_rule.push(rule_fsm.map(|f| {
                let node_num = f.num_states();
                let edge_num = f.fsm().num_edges();
                let (start, ends) = splice_into(&mut complete, &f);
                (start, ends, node_num, edge_num)
            }));
        }

        let compact_complete = Arc::new(complete.to_compact());
        let final_per_rule: Vec<Option<CompactFsmWithStartEnd>> = per_rule
            .into_iter()
            .map(|opt| {
                opt.map(|(start, ends, node_num, edge_num)| {
                    CompactFsmWithStartEnd::new_view(
                        Arc::clone(&compact_complete),
                        start,
                        ends,
                        node_num,
                        edge_num,
                    )
                })
            })
            .collect();

        grammar.complete_fsm = compact_complete;
        grammar.per_rule_fsms = final_per_rule;
    }
}

/// Splice `sub` into `complete`, returning `(new_start, new_ends)` pointing
/// at the spliced-in states without cloning `complete`.
fn splice_into(complete: &mut Fsm, sub: &FsmWithStartEnd) -> (usize, Vec<bool>) {
    let mapping = complete.add_fsm(sub.fsm());
    let new_start = mapping[sub.start()];
    let mut new_ends = vec![false; complete.num_states()];
    for s in 0..sub.num_states() {
        if sub.is_end_state(s) {
            new_ends[mapping[s]] = true;
        }
    }
    (new_start, new_ends)
}

/// FSM for a `RuleRef` element: a single rule-ref edge.
pub fn build_rule_ref(expr: &GrammarExpr<'_>) -> FsmWithStartEnd {
    let mut fsm = FsmWithStartEnd::default();
    fsm.add_state();
    fsm.add_state();
    fsm.set_start_state(0);
    fsm.add_end_state(1);
    fsm.fsm_mut().add_rule_edge(0, 1, expr.data[0] as i16);
    fsm
}

/// FSM for a `Repeat` element: a single repeat-ref edge.
pub fn build_repeat(expr: &GrammarExpr<'_>) -> FsmWithStartEnd {
    let mut fsm = FsmWithStartEnd::default();
    fsm.add_state();
    fsm.add_state();
    fsm.set_start_state(0);
    fsm.add_end_state(1);
    fsm.fsm_mut()
        .add_repeat_edge(0, 1, expr.data[0], expr.data[1], expr.data[2]);
    fsm
}

/// FSM for a `ByteString` element: a linear chain of byte edges.
pub fn build_byte_string(expr: &GrammarExpr<'_>) -> FsmWithStartEnd {
    let mut fsm = FsmWithStartEnd::default();
    let mut cur = fsm.add_state();
    fsm.set_start_state(cur);
    for &byte in expr.data {
        let next = fsm.add_state();
        fsm.fsm_mut().add_edge(cur, next, byte as i16, byte as i16);
        cur = next;
    }
    fsm.add_end_state(cur);
    fsm
}

/// FSM for a `CharacterClass` / `CharacterClassStar` element.
pub fn build_character_class(expr: &GrammarExpr<'_>) -> FsmWithStartEnd {
    let is_negative = expr.data[0] != 0;
    if is_negative {
        return build_negative_character_class(expr);
    }
    let mut fsm = FsmWithStartEnd::default();
    let start = fsm.add_state();
    fsm.set_start_state(start);
    let is_star = expr.kind == GrammarExprType::CharacterClassStar;
    let end = if is_star { start } else { fsm.add_state() };
    fsm.add_end_state(end);
    let mut i = 1;
    while i < expr.data.len() {
        let lo = codepoint_to_packed_utf8(expr.data[i] as u32);
        let hi = codepoint_to_packed_utf8(expr.data[i + 1] as u32);
        add_character_range(&mut fsm, start, end, lo, hi);
        i += 2;
    }
    fsm
}

/// FSM for a negated character class — the complement of the excluded
/// codepoint ranges across `[0x0000, 0x10_FFFF]` (excluding UTF-16
/// surrogates `0xD800..=0xDFFF`).
fn build_negative_character_class(expr: &GrammarExpr<'_>) -> FsmWithStartEnd {
    const MAX_CODEPOINT: u32 = 0x10_FFFF;
    const SURROGATE_LO: u32 = 0xD800;
    const SURROGATE_HI: u32 = 0xDFFF;

    let mut excluded: Vec<(u32, u32)> = Vec::with_capacity(expr.data.len() / 2 + 1);
    excluded.push((SURROGATE_LO, SURROGATE_HI));
    let mut i = 1;
    while i + 1 < expr.data.len() {
        let lo = expr.data[i];
        let hi = expr.data[i + 1];
        if hi >= 0 && lo <= MAX_CODEPOINT as i32 && lo <= hi {
            let clamped_lo = lo.max(0) as u32;
            let clamped_hi = (hi as u32).min(MAX_CODEPOINT);
            excluded.push((clamped_lo, clamped_hi));
        }
        i += 2;
    }
    excluded.sort_unstable();

    let mut merged: Vec<(u32, u32)> = Vec::with_capacity(excluded.len());
    for (lo, hi) in excluded {
        if let Some(last) = merged.last_mut()
            && lo <= last.1.saturating_add(1)
        {
            last.1 = last.1.max(hi);
        } else {
            merged.push((lo, hi));
        }
    }

    let mut fsm = FsmWithStartEnd::default();
    let start = fsm.add_state();
    fsm.set_start_state(start);
    let is_star = expr.kind == GrammarExprType::CharacterClassStar;
    let end = if is_star { start } else { fsm.add_state() };
    fsm.add_end_state(end);

    let mut cursor: u32 = 0;
    for (lo, hi) in merged {
        if cursor < lo {
            let packed_lo = codepoint_to_packed_utf8(cursor);
            let packed_hi = codepoint_to_packed_utf8(lo - 1);
            add_character_range(&mut fsm, start, end, packed_lo, packed_hi);
        }
        if hi >= MAX_CODEPOINT {
            return fsm;
        }
        cursor = hi + 1;
    }
    if cursor <= MAX_CODEPOINT {
        let packed_lo = codepoint_to_packed_utf8(cursor);
        let packed_hi = codepoint_to_packed_utf8(MAX_CODEPOINT);
        add_character_range(&mut fsm, start, end, packed_lo, packed_hi);
    }
    fsm
}

/// FSM for a normalized `Sequence` (a list of element exprs).
/// Returns `None` if any element is not FSM-expressible.
pub fn build_sequence(expr: &GrammarExpr<'_>, grammar: &GrammarData) -> Option<FsmWithStartEnd> {
    let mut parts: Vec<FsmWithStartEnd> = Vec::new();
    for &id in expr.data {
        let e = grammar.expr(id);
        let part = match e.kind {
            GrammarExprType::ByteString => build_byte_string(&e),
            GrammarExprType::RuleRef => build_rule_ref(&e),
            GrammarExprType::CharacterClass | GrammarExprType::CharacterClassStar => {
                build_character_class(&e)
            }
            GrammarExprType::Repeat => build_repeat(&e),
            _ => return None,
        };
        parts.push(part);
    }
    if parts.is_empty() {
        return Some(empty_fsm());
    }
    Some(FsmWithStartEnd::concat(&parts))
}

/// FSM for a normalized rule body — a `Choices` of `Sequence`s (with an
/// optional leading `EmptyStr`). Returns `None` if not FSM-expressible.
pub fn build_choices(expr: &GrammarExpr<'_>, grammar: &GrammarData) -> Option<FsmWithStartEnd> {
    debug_assert_eq!(expr.kind, GrammarExprType::Choices);
    let mut fsm_list: Vec<FsmWithStartEnd> = Vec::new();
    let mut nullable = false;
    for &id in expr.data {
        let e = grammar.expr(id);
        if e.kind == GrammarExprType::EmptyStr {
            nullable = true;
            continue;
        }
        debug_assert_eq!(e.kind, GrammarExprType::Sequence);
        fsm_list.push(build_sequence(&e, grammar)?);
    }
    if fsm_list.is_empty() {
        return Some(empty_fsm());
    }
    if nullable {
        fsm_list.push(empty_fsm());
    }
    let mut result = FsmWithStartEnd::union(&fsm_list);
    result = result.simplify_epsilon();
    result = result.merge_equivalent_successors();
    // Upstream commit 96ae88b (#616) dropped the `MinimizeDFA` call here:
    // Hopcroft minimization does not reduce states on these grammars and
    // scales super-linearly in state count, so it was pure wasted work.
    Some(result)
}

/// A one-state FSM accepting only the empty string.
pub fn empty_fsm() -> FsmWithStartEnd {
    let mut fsm = FsmWithStartEnd::default();
    fsm.add_state();
    fsm.set_start_state(0);
    fsm.add_end_state(0);
    fsm
}

#[cfg(test)]
#[path = "fsm_builder_tests.rs"]
mod tests;
