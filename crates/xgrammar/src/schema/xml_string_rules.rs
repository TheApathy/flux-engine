// SPDX-License-Identifier: AGPL-3.0-only

//! Non-empty XML values that exclude the parameter close tag.
//!
//! Adapted from the Qwen-only repair in donor commit 88bf718f. The KMP
//! automaton omits transitions completing `excluded`. Its non-empty entry
//! rule preserves arbitrary value text without swallowing the close tag.

/// `(rule name, body)` pairs, starting with the non-empty entry rule.
pub(super) fn nonempty_excluding_rules(entry: &str, excluded: &str) -> Vec<(String, String)> {
    let pattern: Vec<char> = excluded.chars().collect();
    assert!(!pattern.is_empty(), "excluded string must be non-empty");
    let len = pattern.len();
    let failure = kmp_failure(&pattern);
    let mut alphabet: Vec<char> = pattern.clone();
    alphabet.sort_unstable();
    alphabet.dedup();

    let delta = |state: usize, c: char| -> usize {
        let mut k = state;
        loop {
            if pattern[k] == c {
                return k + 1;
            }
            if k == 0 {
                return 0;
            }
            k = failure[k - 1];
        }
    };
    let state_name = |s: usize| format!("{entry}_q{s}");
    let other = format!(
        "[^{}]",
        alphabet
            .iter()
            .map(|&c| class_escape(c))
            .collect::<String>()
    );
    let transitions = |s: usize| -> Vec<String> {
        let mut alts = Vec::new();
        for &c in &alphabet {
            let next = delta(s, c);
            if next < len {
                alts.push(format!("\"{}\" {}", literal_escape(c), state_name(next)));
            }
        }
        alts.push(format!("{other} {}", state_name(0)));
        alts
    };

    let mut rules = vec![(entry.to_string(), transitions(0).join(" | "))];
    for s in 0..len {
        let mut alts = vec!["\"\"".to_string()];
        alts.extend(transitions(s));
        rules.push((state_name(s), alts.join(" | ")));
    }
    rules
}

fn kmp_failure(pattern: &[char]) -> Vec<usize> {
    let mut failure = vec![0; pattern.len()];
    let mut k = 0;
    for i in 1..pattern.len() {
        while k > 0 && pattern[i] != pattern[k] {
            k = failure[k - 1];
        }
        if pattern[i] == pattern[k] {
            k += 1;
        }
        failure[i] = k;
    }
    failure
}

fn literal_escape(c: char) -> String {
    match c {
        '"' => "\\\"".to_string(),
        '\\' => "\\\\".to_string(),
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\t' => "\\t".to_string(),
        _ => c.to_string(),
    }
}

fn class_escape(c: char) -> String {
    match c {
        ']' | '\\' | '^' | '-' | '[' => format!("\\{c}"),
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\t' => "\\t".to_string(),
        _ => c.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automaton_rules_cover_every_prefix_state_and_never_complete_the_suffix() {
        let rules = nonempty_excluding_rules("xml_string_nonempty", "</parameter>");
        assert_eq!(rules.len(), 1 + 12);
        assert_eq!(rules[0].0, "xml_string_nonempty");
        assert!(
            !rules[0].1.contains("\"\""),
            "the entry needs one character"
        );
        let last = &rules
            .iter()
            .find(|(n, _)| n == "xml_string_nonempty_q11")
            .unwrap()
            .1;
        assert!(!last.contains("\">\""), "{last}");
        assert!(last.starts_with("\"\""), "{last}");
    }
}
