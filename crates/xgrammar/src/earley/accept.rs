// SPDX-License-Identifier: AGPL-3.0-only
//
// Acceptable-next-byte computation for the Earley parser.
//
// The C++ matcher derives the set of bytes a parser can consume by
// inspecting the outgoing char-range edges / sequence elements of every
// latest scanable state. This module exposes that as a reusable query
// so the W6 `GrammarMatcher` can build its token bitmask on top.

use super::parser::EarleyParser;
use super::state::ParserState;
use crate::grammar::GrammarExprType;

impl EarleyParser {
    /// Append the byte ranges `state` can accept to `ranges` as
    /// inclusive `(min, max)` pairs. FSM-backed states contribute their
    /// char-range edges; non-FSM states contribute the current
    /// element's byte/character-class ranges.
    fn acceptable_ranges_of(&self, state: &ParserState, ranges: &mut Vec<(u8, u8)>) {
        if state.rule_id != -1 {
            let fsm = match self.grammar.per_rule_fsms[state.rule_id as usize].as_ref() {
                Some(f) => f,
                None => return,
            };
            for edge in fsm.fsm().edges(state.element_id as usize) {
                if edge.is_char_range() {
                    ranges.push((edge.min as u8, edge.max as u8));
                }
            }
            return;
        }
        let seq = self.grammar.expr(state.sequence_id);
        if state.element_id as usize >= seq.len() {
            return;
        }
        let element = self.grammar.expr(seq[state.element_id as usize]);
        match element.kind {
            GrammarExprType::ByteString if (state.sub_element_id as usize) < element.len() => {
                let b = element[state.sub_element_id as usize] as u8;
                ranges.push((b, b));
            }
            GrammarExprType::CharacterClass | GrammarExprType::CharacterClassStar => {
                if state.sub_element_id > 0 {
                    // Mid-UTF-8 multi-byte character: only continuation bytes are valid.
                    ranges.push((0x80, 0xBF));
                    return;
                }
                let is_negative = element[0] != 0;
                if is_negative {
                    let mut excluded_ascii = [false; 128];
                    let mut excludes_all_non_ascii = false;
                    let mut i = 1;
                    while i + 1 < element.len() {
                        let lo = element[i];
                        let hi = element[i + 1];
                        if lo <= hi {
                            if lo <= 127 && hi >= 0 {
                                let a_lo = lo.max(0) as usize;
                                let a_hi = hi.min(127) as usize;
                                excluded_ascii[a_lo..=a_hi].fill(true);
                            }
                            if lo <= 0x80 && hi >= 0x10_FFFF {
                                excludes_all_non_ascii = true;
                            }
                        }
                        i += 2;
                    }
                    let mut lo = 0usize;
                    while lo < 128 {
                        if excluded_ascii[lo] {
                            lo += 1;
                            continue;
                        }
                        let mut hi = lo + 1;
                        while hi < 128 && !excluded_ascii[hi] {
                            hi += 1;
                        }
                        ranges.push((lo as u8, (hi - 1) as u8));
                        lo = hi;
                    }
                    if !excludes_all_non_ascii {
                        ranges.push((0xC2, 0xF4));
                    }
                } else {
                    let mut i = 1;
                    while i + 1 < element.len() {
                        let lo = element[i];
                        let hi = element[i + 1];
                        if lo <= hi {
                            if lo <= 127 && hi >= 0 {
                                ranges.push((lo.max(0) as u8, hi.min(127) as u8));
                            }
                            if hi >= 0x80 && lo <= 0x10_FFFF {
                                let u_lo = lo.max(0x80) as u32;
                                let u_hi = hi.min(0x10_FFFF) as u32;
                                ranges.push((utf8_lead_byte(u_lo), utf8_lead_byte(u_hi)));
                            }
                        }
                        i += 2;
                    }
                }
            }
            _ => {}
        }
    }

    /// All byte ranges acceptable as the next input, across every
    /// latest scanable state. Ranges may overlap and are unsorted.
    pub fn acceptable_byte_ranges(&self) -> Vec<(u8, u8)> {
        let mut ranges = Vec::new();
        for state in self.latest_scanable_states() {
            self.acceptable_ranges_of(state, &mut ranges);
        }
        ranges
    }

    /// A 256-entry bitmap: `out[b]` is true if byte `b` can be the next
    /// input. Exact for FSM-backed states (every optimized grammar); for
    /// non-FSM multi-byte character classes this is a conservative byte-level
    /// pre-check and `advance` is authoritative.
    pub fn acceptable_byte_mask(&self) -> [bool; 256] {
        let mut mask = [false; 256];
        for (lo, hi) in self.acceptable_byte_ranges() {
            for b in lo..=hi {
                mask[b as usize] = true;
            }
        }
        mask
    }

    /// True if byte `ch` can be accepted as the next input without
    /// mutating the parser. Exact for FSM-backed states; for non-FSM
    /// multi-byte character classes this is a conservative pre-check and
    /// `advance` is authoritative.
    pub fn can_accept(&self, ch: u8) -> bool {
        self.acceptable_byte_mask()[ch as usize]
    }
}

/// First UTF-8 byte of a non-ASCII Unicode codepoint in `0x80..=0x10_FFFF`.
/// Monotonically non-decreasing across that domain (`0xC2..=0xF4`).
fn utf8_lead_byte(cp: u32) -> u8 {
    if cp <= 0x7FF {
        (0xC0 | (cp >> 6)) as u8
    } else if cp <= 0xFFFF {
        (0xE0 | (cp >> 12)) as u8
    } else {
        (0xF0 | (cp >> 18)) as u8
    }
}
