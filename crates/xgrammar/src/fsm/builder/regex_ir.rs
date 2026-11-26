// SPDX-License-Identifier: AGPL-3.0-only
//
// RegexIr — the intermediate representation a regex string is parsed
// into, plus its NFA construction (`Build` / `visit`). Port of `RegexIR`
// from `cpp/fsm_builder.cc`.
//
// The C++ uses `std::variant`; Rust uses a plain enum. Leaf FSMs are
// built directly from regex fragments via `build_leaf_fsm`.

use crate::fsm::fsm::Fsm;
use crate::fsm::with_start_end::FsmWithStartEnd;

/// Sentinel for an unbounded repeat upper bound (`{n,}`).
pub const REPEAT_NO_UPPER_BOUND: i32 = -1;

/// A postfix-style regex quantifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegexSymbol {
    /// `*`
    Star,
    /// `+`
    Plus,
    /// `?`
    Optional,
}

/// A node of the regex IR tree.
#[derive(Debug, Clone)]
pub enum RegexState {
    /// A literal fragment or character class, e.g. `"a"` or `"[a-z]"`.
    /// Stored as raw bytes — the regex grammar is byte-wise, so a
    /// fragment may split a UTF-8 codepoint across multiple `Leaf`s.
    Leaf { regex: Vec<u8> },
    /// A quantifier applied to a single child.
    Symbol {
        symbol: RegexSymbol,
        state: Box<RegexState>,
    },
    /// An alternation of branches.
    Union { states: Vec<RegexState> },
    /// A parenthesized concatenation.
    Bracket { states: Vec<RegexState> },
    /// A counted repetition `{lower,upper}`.
    Repeat {
        states: Vec<RegexState>,
        lower_bound: i32,
        upper_bound: i32,
    },
}

/// The whole parsed regex: a sequence of top-level states.
#[derive(Debug, Default, Clone)]
pub struct RegexIr {
    /// Top-level concatenation of states.
    pub states: Vec<RegexState>,
}

impl RegexIr {
    /// Build the NFA for the entire IR.
    pub fn build(&self) -> Result<FsmWithStartEnd, String> {
        if self.states.is_empty() {
            // The empty regex accepts only the empty string.
            let empty = Fsm::with_states(1);
            return Ok(FsmWithStartEnd::new(empty, 0, vec![true], false));
        }
        let mut fsm_list: Vec<FsmWithStartEnd> = Vec::new();
        for state in &self.states {
            fsm_list.push(visit(state)?);
        }
        if fsm_list.len() > 1 {
            Ok(FsmWithStartEnd::concat(&fsm_list))
        } else {
            Ok(fsm_list.into_iter().next().unwrap())
        }
    }
}

/// Construct the NFA for a single IR node.
pub fn visit(state: &RegexState) -> Result<FsmWithStartEnd, String> {
    match state {
        RegexState::Leaf { regex } => Ok(build_leaf_fsm(regex)),
        // (see build_leaf_fsm — takes a &[u8])
        RegexState::Symbol { symbol, state } => {
            let child = visit(state)?;
            Ok(match symbol {
                RegexSymbol::Plus => child.plus(),
                RegexSymbol::Star => child.star(),
                RegexSymbol::Optional => child.optional(),
            })
        }
        RegexState::Union { states } => {
            let mut list = Vec::new();
            for child in states {
                list.push(visit(child)?);
            }
            if list.len() <= 1 {
                return Err("Invalid union".to_string());
            }
            Ok(FsmWithStartEnd::union(&list))
        }
        RegexState::Bracket { states } => {
            let mut list = Vec::new();
            for child in states {
                list.push(visit(child)?);
            }
            if list.is_empty() {
                return Err("Invalid bracket".to_string());
            }
            Ok(FsmWithStartEnd::concat(&list))
        }
        RegexState::Repeat {
            states,
            lower_bound,
            upper_bound,
        } => {
            if states.len() != 1 {
                return Err("Invalid repeat".to_string());
            }
            visit_repeat(&states[0], *lower_bound, *upper_bound)
        }
    }
}

/// NFA construction for a `{lower,upper}` repetition.
fn visit_repeat(
    child_state: &RegexState,
    lower_bound: i32,
    upper_bound: i32,
) -> Result<FsmWithStartEnd, String> {
    if lower_bound < 0 || (upper_bound != REPEAT_NO_UPPER_BOUND && lower_bound > upper_bound) {
        return Err(format!(
            "Invalid repeat bounds: {{{lower_bound},{upper_bound}}}"
        ));
    }
    let child = visit(child_state)?;

    if upper_bound == 0 {
        return Ok(FsmWithStartEnd::new(
            Fsm::with_states(1),
            0,
            vec![true],
            false,
        ));
    }

    if upper_bound == REPEAT_NO_UPPER_BOUND {
        return Ok(match lower_bound {
            0 => child.star(),
            1 => child.plus(),
            n => {
                let mut parts: Vec<FsmWithStartEnd> = Vec::with_capacity(n as usize);
                for _ in 0..(n - 1) {
                    parts.push(child.clone());
                }
                parts.push(child.plus());
                FsmWithStartEnd::concat(&parts)
            }
        });
    }

    // Finite {lower,upper} with 0 <= lower <= upper and upper >= 1:
    // child^{lower} followed by (child?)^{upper - lower}.
    let mut parts: Vec<FsmWithStartEnd> = Vec::with_capacity(upper_bound as usize);
    for _ in 0..lower_bound {
        parts.push(child.clone());
    }
    if upper_bound > lower_bound {
        let opt = child.optional();
        for _ in lower_bound..upper_bound {
            parts.push(opt.clone());
        }
    }
    Ok(FsmWithStartEnd::concat(&parts))
}

// Leaf-FSM construction lives in `regex_leaf.rs`; re-exported here
// so existing call sites (`super::regex_ir::build_leaf_fsm`) keep
// working.
pub use super::regex_leaf::{build_leaf_fsm, handle_escapes};

#[cfg(test)]
#[path = "regex_ir_tests.rs"]
mod tests;
