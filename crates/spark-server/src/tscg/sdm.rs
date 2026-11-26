// SPDX-License-Identifier: AGPL-3.0-only
//
// TSCG operator SDM — Semantic Density Maximization.
//
// Maximises information per token by deleting filler that carries no
// schema-relevant meaning: politeness markers, hedging, redundant
// connectives, and meta-phrases. The paper lists 104+ patterns; this
// port carries the high-value subset that recurs in real tool
// descriptions (Anthropic / OpenAI / opencode tool catalogues).
//
// SDM is whitespace- and case-preserving except where a pattern is
// deleted; it never reorders or paraphrases, so it cannot change a
// description's meaning — it only removes provably-empty spans.

/// Filler phrases deleted outright (ASCII case-insensitive, whole-substring).
/// Ordered longest-first so a longer match wins over a shorter prefix.
const FILLER: &[&str] = &[
    "it is important to note that ",
    "it should be noted that ",
    "please be aware that ",
    "as a general rule, ",
    "for the purpose of ",
    "where appropriate, ",
    "please note that ",
    "with the goal of ",
    "you can use this ",
    "can be used to ",
    "if applicable, ",
    "this function ",
    "allows you to ",
    "the following ",
    "if necessary, ",
    "in order to ",
    "this method ",
    "use this to ",
    "one or more ",
    "essentially ",
    "is used to ",
    "if needed, ",
    "as needed, ",
    "this tool ",
    "helps you ",
    "a list of ",
    "basically ",
    "lets you ",
    "used to ",
    "kindly ",
    "simply ",
    "note: ",
    "just ",
];

/// Verbose → compact phrase rewrites (DRO-adjacent prose compaction).
/// Applied after filler removal. Case-insensitive on the key.
const REWRITES: &[(&str, &str)] = &[
    ("corresponds to", "→"),
    ("for example", "e.g."),
    ("for instance", "e.g."),
    ("such as", "e.g."),
    ("and so on", "etc."),
    ("optional ", ""),
    ("a string representing ", ""),
    ("a string containing ", ""),
    ("an integer representing ", ""),
    ("the name of the ", ""),
    ("the path to the ", "path: "),
    ("must be ", ""),
    ("should be ", ""),
];

/// Compress a description string. Returns a single-line, filler-free,
/// whitespace-collapsed form. Empty input → empty output.
pub fn densify(text: &str) -> String {
    if text.trim().is_empty() {
        return String::new();
    }
    // Collapse newlines/tabs to spaces first so multi-line descriptions
    // become one line (the TSCG block is line-structured).
    let mut s: String = text
        .chars()
        .map(|c| {
            if c == '\n' || c == '\t' || c == '\r' {
                ' '
            } else {
                c
            }
        })
        .collect();

    // Filler removal — ASCII lower-cased scan (1:1 byte-preserving), splice on
    // the original string so casing of surviving text and non-ASCII UTF-8
    // offsets are preserved.
    for pat in FILLER {
        loop {
            let lower = s.to_ascii_lowercase();
            match lower.find(pat) {
                Some(idx) => {
                    s.replace_range(idx..idx + pat.len(), "");
                }
                None => break,
            }
        }
    }

    // Phrase rewrites (all keys are pure ASCII, so `to_ascii_lowercase` keeps
    // byte offsets identical to `s`).
    for (from, to) in REWRITES {
        loop {
            let lower = s.to_ascii_lowercase();
            match lower.find(from) {
                Some(idx) => {
                    s.replace_range(idx..idx + from.len(), to);
                }
                None => break,
            }
        }
    }

    // Collapse runs of whitespace and trim leading/trailing spaces.
    let collapsed: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_filler() {
        assert_eq!(
            densify("This tool allows you to search the project files"),
            "search the project files"
        );
    }

    #[test]
    fn filler_ordered_longest_first_and_ascii() {
        assert!(
            FILLER.windows(2).all(|w| w[0].len() >= w[1].len()),
            "FILLER must be ordered by descending byte length"
        );
        assert!(FILLER.iter().all(|p| p.is_ascii()));
        assert!(REWRITES.iter().all(|(from, _)| from.is_ascii()));
        assert_eq!(densify("can be used to search files"), "search files");
        assert_eq!(densify("is used to inspect logs"), "inspect logs");
    }

    #[test]
    fn non_ascii_case_folding_does_not_panic() {
        // U+0130 ('İ') and U+212A ('K') change UTF-8 byte length under
        // Unicode `to_lowercase()`, which must never misalign splice offsets.
        assert_eq!(
            densify("İ can be used to search K for example"),
            "İ search K e.g."
        );
    }

    #[test]
    fn collapses_multiline() {
        assert_eq!(
            densify("line one\n  line two\n\tline three"),
            "line one line two line three"
        );
    }

    #[test]
    fn empty_stays_empty() {
        assert_eq!(densify(""), "");
        assert_eq!(densify("   \n  "), "");
    }

    #[test]
    fn preserves_meaningful_casing() {
        // "Bash" must survive — only the filler prefix is removed.
        assert_eq!(
            densify("Use this to run Bash commands"),
            "run Bash commands"
        );
    }
}
