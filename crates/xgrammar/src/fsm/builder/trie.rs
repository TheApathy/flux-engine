// SPDX-License-Identifier: AGPL-3.0-only
//
// TrieFsmBuilder — builds a trie / Aho-Corasick FSM from a pattern list.
// Port of `TrieFSMBuilder` / `TrieFSMBuilderImpl` from `cpp/fsm_builder.cc`.

use ahash::AHashSet;
use std::collections::VecDeque;

use crate::fsm::edge::{FsmEdge, edge_type};
use crate::fsm::fsm::{Fsm, NO_NEXT_STATE};
use crate::fsm::with_start_end::FsmWithStartEnd;

/// Result of a trie build — the FSM plus the per-pattern terminal states.
pub struct TrieBuildResult {
    /// The constructed FSM with start/end states.
    pub fsm: FsmWithStartEnd,
    /// Terminal state of each input pattern, in input order.
    pub end_states: Vec<i32>,
}

/// Build a trie-based FSM.
///
/// `patterns` are inserted as a character trie. `excluded_patterns` are
/// only honored when `add_back_edges` is true (the trie becomes an
/// Aho-Corasick automaton with the excluded terminals pruned).
///
/// Returns `None` when `allow_overlap` is false and a pattern is empty
/// or is a prefix of / shares a terminal with another pattern.
pub fn build_trie(
    patterns: &[&[u8]],
    excluded_patterns: &[&[u8]],
    allow_overlap: bool,
    add_back_edges: bool,
) -> Option<TrieBuildResult> {
    let mut fsm = Fsm::with_states(1);
    let start = 0usize;
    let mut ends: AHashSet<i32> = AHashSet::new();
    let mut end_states: Vec<i32> = Vec::new();

    for pattern in patterns {
        if !allow_overlap && pattern.is_empty() {
            return None;
        }
        let mut current = 0i32;
        for &ch in *pattern {
            let ch16 = ch as i16;
            let mut next = fsm.next_state(current as usize, ch16 as i32, edge_type::CHAR_RANGE);
            if next == NO_NEXT_STATE {
                next = fsm.add_state() as i32;
                fsm.add_edge(current as usize, next as usize, ch16, ch16);
            }
            current = next;
            if !allow_overlap && ends.contains(&current) {
                return None;
            }
        }
        if !allow_overlap && !fsm.edges(current as usize).is_empty() {
            return None;
        }
        ends.insert(current);
        end_states.push(current);
    }

    let mut dead_states: AHashSet<i32> = AHashSet::new();

    if add_back_edges {
        for excluded in excluded_patterns {
            if !allow_overlap && excluded.is_empty() {
                return None;
            }
            let mut current = 0i32;
            for &ch in *excluded {
                let ch16 = ch as i16;
                let mut next = fsm.next_state(current as usize, ch16 as i32, edge_type::CHAR_RANGE);
                if next == NO_NEXT_STATE {
                    next = fsm.add_state() as i32;
                    fsm.add_edge(current as usize, next as usize, ch16, ch16);
                }
                current = next;
                if !allow_overlap && ends.contains(&current) {
                    return None;
                }
            }
            if !allow_overlap && !fsm.edges(current as usize).is_empty() {
                return None;
            }
            ends.insert(current);
            dead_states.insert(current);
        }

        add_back_edges_to_fsm(&mut fsm, start, &ends);

        // Drop edges pointing at excluded (dead) terminal states.
        if !dead_states.is_empty() {
            for state in 0..fsm.num_states() {
                let kept: Vec<FsmEdge> = fsm
                    .edges(state)
                    .iter()
                    .copied()
                    .filter(|e| !dead_states.contains(&e.target))
                    .collect();
                *fsm.edges_mut(state) = kept;
            }
        }
    }

    let mut is_end_state = vec![false; fsm.num_states()];
    for &end in &ends {
        if !dead_states.contains(&end) {
            is_end_state[end as usize] = true;
        }
    }
    Some(TrieBuildResult {
        fsm: FsmWithStartEnd::new(fsm, start, is_end_state, false),
        end_states,
    })
}

/// Insert Aho-Corasick back edges so a failed match restarts from the
/// longest matching suffix (standard BFS failure links).
fn add_back_edges_to_fsm(fsm: &mut Fsm, start: usize, ends: &AHashSet<i32>) {
    let n = fsm.num_states();
    // Snapshot the pure single-byte trie transitions before adding back edges.
    let trie_edges: Vec<Vec<(u8, usize)>> = (0..n)
        .map(|u| {
            let mut v: Vec<(u8, usize)> = fsm
                .edges(u)
                .iter()
                .filter(|e| e.is_char_range() && e.min == e.max)
                .map(|e| (e.min as u8, e.target as usize))
                .collect();
            v.sort_unstable_by_key(|&(c, _)| c);
            v
        })
        .collect();
    let trie_next = |u: usize, c: u8| -> Option<usize> {
        trie_edges[u]
            .binary_search_by_key(&c, |&(ch, _)| ch)
            .ok()
            .map(|idx| trie_edges[u][idx].1)
    };

    // Standard BFS construction of Aho-Corasick failure links (`fail[u]`).
    let mut fail = vec![start; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for &(_c, v) in &trie_edges[start] {
        fail[v] = start;
        queue.push_back(v);
    }
    while let Some(u) = queue.pop_front() {
        for &(c, v) in &trie_edges[u] {
            let mut f = fail[u];
            while f != start && trie_next(f, c).is_none() {
                f = fail[f];
            }
            fail[v] = trie_next(f, c).unwrap_or(start);
            queue.push_back(v);
        }
    }

    // Complete transitions for `start` and every non-terminal state by
    // overlaying trie transitions along the failure chain `start -> ... -> u`,
    // then coalescing adjacent bytes sharing the same target into maximal ranges.
    let mut chain: Vec<usize> = Vec::new();
    for u in 0..n {
        if u != start && ends.contains(&(u as i32)) {
            continue;
        }
        chain.clear();
        let mut cur = u;
        loop {
            chain.push(cur);
            if cur == start {
                break;
            }
            cur = fail[cur];
        }
        let mut target = [start as i32; 256];
        for &node in chain.iter().rev() {
            for &(c, next) in &trie_edges[node] {
                target[c as usize] = next as i32;
            }
        }

        let mut new_edges: Vec<FsmEdge> = Vec::new();
        let mut lo = 0usize;
        while lo < 256 {
            let t = target[lo];
            let mut hi = lo + 1;
            while hi < 256 && target[hi] == t {
                hi += 1;
            }
            new_edges.push(FsmEdge::new(lo as i16, (hi - 1) as i16, t));
            lo = hi;
        }
        *fsm.edges_mut(u) = new_edges;
    }
}

#[cfg(test)]
#[path = "trie_tests.rs"]
mod tests;
