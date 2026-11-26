// SPDX-License-Identifier: AGPL-3.0-only

//! CFG jump-forward mismatched-closer repair for code-generation drafting
//! (`ATLAS_DFLASH_CFG_JF=1`, default OFF).
//!
//! # What this does
//!
//! This module maintains a lightweight per-splice **bracket / string tracker**
//! derived from the committed token stream. When the neural drafter explicitly
//! proposes a closing bracket (`)`, `]`, or `}`) outside a string literal that
//! **contradicts** the innermost open bracket on the tracker stack, it replaces
//! that mismatched closer in place with the matching closing bracket token ID.
//!
//! It does **not** force closers over ordinary content tokens, opening
//! delimiters, or string contents, does **not** perform colon/newline forcing,
//! and never truncates `drafts` (so the verify graph width is preserved).
//!
//! # Why it is LOSSLESS by construction
//!
//! Splicing only changes WHAT is proposed. The verify path commits solely the
//! target's greedy argmax token and accepts a draft *only* when
//! `draft == target_argmax`. A wrong splice is therefore rejected exactly like
//! a wrong drafter token and can NEVER change committed output. With the flag
//! off, drafts pass through untouched.
//!
//! # Design: token-id classification, not string re-decode on the hot path
//!
//! Every vocabulary id can be classified once at startup ([`build_delim_table`])
//! into a compact [`TokenDelim`] and cached in a process-wide table. The hot
//! path performs one table lookup per token — no tokenizer, no allocation.
//!
//! # Honest limitations (when unsure, DON'T splice)
//!
//! * Strings / comments / f-strings: a `)` inside a string literal is literal
//!   text, not a structural close. The tracker suppresses bracket bookkeeping
//!   inside single/double/triple quotes. When either the committed-prefix walk
//!   or the draft walk hits an ambiguous (`TokenDelim::Opaque`) token or a
//!   mismatched close in committed text, the tracker becomes poisoned and
//!   disables splicing for that call rather than guessing.
//! * Only mismatched closing brackets (`)` / `]` / `}`) outside strings are
//!   ever replaced.

use std::sync::OnceLock;

/// Structural classification of a single vocabulary token, derived from its
/// decoded string. Only delimiter-relevant classes are distinguished; every
/// other token is [`TokenDelim::Other`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenDelim {
    /// Token is exactly one opening bracket: `(`, `[`, or `{`.
    Open(u8),
    /// Token is exactly one closing bracket: `)`, `]`, or `}`.
    Close(u8),
    /// Token is exactly a single-line quote delimiter char: `'` or `"` or `` ` ``.
    Quote(u8),
    /// Token is exactly a triple-quote: `'''` or `"""`.
    TripleQuote(u8),
    /// Token is exactly `:` (colon).
    Colon,
    /// Token is pure whitespace containing a `\n`.
    Newline,
    /// A normal content token: neither structural nor an ambiguous multi-symbol token.
    Other,
    /// The token's decoded string is structurally AMBIGUOUS — it mixes a quote
    /// with a bracket, contains an unbalanced set of delimiters we can't reason
    /// about, or failed to decode. When the tracker meets an opaque token
    /// (in either the committed stream or a draft) it poisons and refuses to
    /// splice. Fail-safe: opaque never *forces* anything.
    Opaque,
}

/// Match a closing bracket byte to its opening counterpart.
fn matching_open(close: u8) -> u8 {
    match close {
        b')' => b'(',
        b']' => b'[',
        b'}' => b'{',
        other => other,
    }
}

/// Match an opening bracket byte to its closing counterpart.
fn matching_close(open: u8) -> u8 {
    match open {
        b'(' => b')',
        b'[' => b']',
        b'{' => b'}',
        other => other,
    }
}

/// Classify a single token's decoded string into a [`TokenDelim`].
///
/// The classification is deliberately CONSERVATIVE: only tokens whose string is
/// *exactly* one recognized delimiter (optionally with surrounding horizontal
/// whitespace) get a structural class. Anything that packs a delimiter together
/// with other content (`");"`, `"()"`, `"':'"`, `"](x"`) is [`TokenDelim::Opaque`]
/// — an opaque token poisons the tracker and halts splicing.
pub fn classify_str(s: &str) -> TokenDelim {
    let has_newline = s.contains('\n');
    let core = s.trim_matches([' ', '\t', '\r']);

    if has_newline {
        let non_ws: String = core.chars().filter(|c| !c.is_whitespace()).collect();
        if non_ws.is_empty() {
            return TokenDelim::Newline;
        }
        return TokenDelim::Opaque;
    }

    match core {
        "(" => TokenDelim::Open(b'('),
        "[" => TokenDelim::Open(b'['),
        "{" => TokenDelim::Open(b'{'),
        ")" => TokenDelim::Close(b')'),
        "]" => TokenDelim::Close(b']'),
        "}" => TokenDelim::Close(b'}'),
        "'" => TokenDelim::Quote(b'\''),
        "\"" => TokenDelim::Quote(b'"'),
        "`" => TokenDelim::Quote(b'`'),
        "'''" => TokenDelim::TripleQuote(b'\''),
        "\"\"\"" => TokenDelim::TripleQuote(b'"'),
        ":" => TokenDelim::Colon,
        "" => TokenDelim::Other,
        _ => {
            if core.bytes().any(|b| {
                matches!(
                    b,
                    b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'\'' | b'"' | b'`' | b':'
                )
            }) {
                TokenDelim::Opaque
            } else {
                TokenDelim::Other
            }
        }
    }
}

/// Process-wide token→[`TokenDelim`] table, indexed by token id. `None` until
/// [`set_delim_table`] runs. Absent ⇒ CFG jump-forward is inert.
static DELIM_TABLE: OnceLock<std::sync::Arc<[TokenDelim]>> = OnceLock::new();

/// Install the classification table. Idempotent (first writer wins).
pub fn set_delim_table(table: std::sync::Arc<[TokenDelim]>) {
    let _ = DELIM_TABLE.set(table);
}

/// Read the classification table. `None` until [`set_delim_table`] runs.
pub fn delim_table() -> Option<std::sync::Arc<[TokenDelim]>> {
    DELIM_TABLE.get().cloned()
}

/// Whether CFG jump-forward is enabled (`ATLAS_DFLASH_CFG_JF=1`). Read once.
pub fn cfg_jf_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DFLASH_CFG_JF").ok().as_deref() == Some("1"))
}

/// Build the classification table from the tokenizer by decoding every id.
///
/// `decode` is the tokenizer's single-id decode (byte-level, so a leading BPE
/// space is a real `' '`). Any decode error leaves that id [`TokenDelim::Other`].
/// Returns the table plus the count of ids that got a structural (non-Other,
/// non-Opaque) class, for logging.
pub fn build_delim_table<F>(vocab_size: usize, mut decode: F) -> (Vec<TokenDelim>, usize)
where
    F: FnMut(u32) -> Option<String>,
{
    let mut table = vec![TokenDelim::Other; vocab_size];
    let mut structural = 0usize;
    for (id, slot) in table.iter_mut().enumerate() {
        if let Some(s) = decode(id as u32) {
            let c = classify_str(&s);
            *slot = c;
            if !matches!(c, TokenDelim::Other | TokenDelim::Opaque) {
                structural += 1;
            }
        }
    }
    (table, structural)
}

/// A single open-delimiter frame on the tracker stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frame {
    /// An open bracket; carries its opening byte so we know the matching closer.
    Bracket(u8),
    /// An open single-line string; carries its quote byte.
    Str(u8),
    /// An open triple-quoted string; carries its quote byte.
    TripleStr(u8),
}

/// Bracket / string tracker. Walks a token stream left-to-right maintaining a
/// stack of open brackets and quotes.
#[derive(Clone, Debug, Default)]
pub struct BracketTracker {
    stack: Vec<Frame>,
    /// Set whenever either the committed-prefix walk or the draft walk hits an
    /// opaque token or an unexpected/mismatched closing bracket, disabling
    /// splicing fail-safely for the remainder of the call.
    poisoned: bool,
}

impl BracketTracker {
    /// Fresh tracker with an empty stack.
    pub fn new() -> Self {
        Self::default()
    }

    /// True while inside any string (single or triple). Bracket bookkeeping is
    /// suppressed here — a `)` inside a string is literal text.
    fn in_string(&self) -> bool {
        matches!(self.stack.last(), Some(Frame::Str(_) | Frame::TripleStr(_)))
    }

    /// Advance the tracker by one classified token. Pure state mutation; does
    /// not decide splices. `poisoned` short-circuits to a no-op so a walk that
    /// gave up stays given-up.
    pub fn advance(&mut self, d: TokenDelim) {
        if self.poisoned {
            return;
        }
        match d {
            TokenDelim::TripleQuote(q) => match self.stack.last() {
                Some(Frame::TripleStr(open)) if *open == q => {
                    self.stack.pop();
                }
                Some(Frame::Str(_)) => {}
                _ => self.stack.push(Frame::TripleStr(q)),
            },
            TokenDelim::Quote(q) => match self.stack.last() {
                Some(Frame::Str(open)) if *open == q => {
                    self.stack.pop();
                }
                Some(Frame::Str(_)) | Some(Frame::TripleStr(_)) => {}
                _ => self.stack.push(Frame::Str(q)),
            },
            _ if self.in_string() => {}
            TokenDelim::Open(b) => {
                self.stack.push(Frame::Bracket(b));
            }
            TokenDelim::Close(b) => {
                let want = matching_open(b);
                match self.stack.last() {
                    Some(Frame::Bracket(open)) if *open == want => {
                        self.stack.pop();
                    }
                    _ => self.poisoned = true,
                }
            }
            TokenDelim::Colon | TokenDelim::Newline | TokenDelim::Other => {}
            TokenDelim::Opaque => {
                self.poisoned = true;
            }
        }
    }

    /// Return a replacement closer ONLY when the drafter explicitly proposed a
    /// closing bracket (`)`, `]`, or `}`) outside a string literal that does
    /// not match the innermost open bracket on the stack.
    ///
    /// Never overwrites ordinary content tokens, opening brackets, or anything
    /// inside a string literal, and never fires when `poisoned`.
    fn forced_replacement(&self, drafted: TokenDelim) -> Option<TokenDelim> {
        if self.poisoned || self.in_string() {
            return None;
        }
        let TokenDelim::Close(got) = drafted else {
            return None;
        };
        let Some(Frame::Bracket(open)) = self.stack.last() else {
            return None;
        };
        let expected = matching_close(*open);
        (got != expected).then_some(TokenDelim::Close(expected))
    }
}

/// Concrete bare token IDs for closing brackets (`)`, `]`, `}`), built directly
/// from decoded vocabulary strings in [`build_forced_ids`] (independent of the
/// [`TokenDelim`] classification table) so spliced closers never inject leading
/// whitespace.
#[derive(Clone, Debug, Default)]
pub struct ForcedIds {
    /// First exact-bare id for `)` / `]` / `}` keyed by closing byte.
    close: [Option<u32>; 3],
}

fn close_idx(b: u8) -> Option<usize> {
    match b {
        b')' => Some(0),
        b']' => Some(1),
        b'}' => Some(2),
        _ => None,
    }
}

impl ForcedIds {
    /// Resolve a replacement [`TokenDelim::Close`] to its exact-bare token id,
    /// if the vocabulary contains one.
    fn id_for(&self, d: TokenDelim) -> Option<u32> {
        match d {
            TokenDelim::Close(b) => close_idx(b).and_then(|i| self.close[i]),
            _ => None,
        }
    }
}

/// Build the reverse map from closing bracket to an exact-bare token id by
/// scanning decoded vocabulary strings directly (ignoring [`TokenDelim`]
/// whitespace-trimmed variants so a splice never alters indentation).
pub fn build_forced_ids<F>(vocab_size: usize, mut decode: F) -> ForcedIds
where
    F: FnMut(u32) -> Option<String>,
{
    let mut ids = ForcedIds::default();
    for id in 0..vocab_size as u32 {
        let Some(s) = decode(id) else { continue };
        match s.as_str() {
            ")" => set_first(&mut ids.close[0], id),
            "]" => set_first(&mut ids.close[1], id),
            "}" => set_first(&mut ids.close[2], id),
            _ => {}
        }
    }
    ids
}

fn set_first(slot: &mut Option<u32>, id: u32) {
    if slot.is_none() {
        *slot = Some(id);
    }
}

/// Process-wide forced-id map, built alongside [`set_delim_table`].
static FORCED_IDS: OnceLock<std::sync::Arc<ForcedIds>> = OnceLock::new();

/// Install the forced-id map. Idempotent.
pub fn set_forced_ids(ids: std::sync::Arc<ForcedIds>) {
    let _ = FORCED_IDS.set(ids);
}

/// Read the forced-id map. `None` until [`set_forced_ids`] runs.
pub fn forced_ids() -> Option<std::sync::Arc<ForcedIds>> {
    FORCED_IDS.get().cloned()
}

/// Outcome of one splice pass, for the one-shot stats log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpliceStats {
    /// Number of draft positions whose mismatched closing bracket was replaced
    /// with the innermost open bracket's matching closer.
    pub splices: usize,
    /// The compact position of the first splice (for the stats log), or
    /// `usize::MAX` if none.
    pub first_pos: usize,
}

/// Repair mismatched closing brackets in `drafts` using the bracket stack built
/// from the last 4,096 tokens of `committed`.
///
/// Walks the bounded committed suffix first; if the committed prefix hits an
/// opaque token or mismatched close, `base.poisoned` becomes `true` and this
/// call returns immediately with zero splices (fail-safe).
///
/// Otherwise walks `drafts` left-to-right:
/// * If the drafter proposes a closing bracket outside a string literal that
///   mismatches the innermost open bracket and `forced` has a bare token id for
///   the matching closer, replaces `*slot` in place and advances the tracker by
///   the repaired closer.
/// * Otherwise keeps `*slot` untouched and advances the tracker by it (an
///   opaque token or unmatched close with no open bracket poisons the tracker
///   and stops splicing for the rest of the chain).
///
/// Mutates `drafts` in place without changing `drafts.len()`.
pub fn splice_forced(
    drafts: &mut [u32],
    committed: &[u32],
    table: &[TokenDelim],
    forced: &ForcedIds,
) -> SpliceStats {
    let mut stats = SpliceStats {
        splices: 0,
        first_pos: usize::MAX,
    };
    let mut base = BracketTracker::new();
    let start = committed.len().saturating_sub(4096);
    for &tok in &committed[start..] {
        let d = table
            .get(tok as usize)
            .copied()
            .unwrap_or(TokenDelim::Other);
        base.advance(d);
        if base.poisoned {
            return stats;
        }
    }

    let mut trk = base;
    let mut stopped = false;
    for (pos, slot) in drafts.iter_mut().enumerate() {
        if stopped || trk.poisoned {
            break;
        }
        let drafted = *slot;
        let drafted_delim = table
            .get(drafted as usize)
            .copied()
            .unwrap_or(TokenDelim::Other);

        if let Some(forced_delim) = trk.forced_replacement(drafted_delim) {
            if let Some(fid) = forced.id_for(forced_delim) {
                *slot = fid;
                stats.splices += 1;
                if stats.first_pos == usize::MAX {
                    stats.first_pos = pos;
                }
                trk.advance(forced_delim);
                continue;
            }
            trk.advance(drafted_delim);
            stopped = true;
            continue;
        }

        trk.advance(drafted_delim);
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_vocab() -> Vec<&'static str> {
        vec![
            "(",      // 0
            ")",      // 1
            "[",      // 2
            "]",      // 3
            "{",      // 4
            "}",      // 5
            "\"",     // 6
            "'",      // 7
            ":",      // 8
            "\n",     // 9
            "foo",    // 10
            "bar",    // 11
            "\"\"\"", // 12
            "(x",     // 13 opaque
            "  ",     // 14 whitespace-only -> Other
            " (",     // 15 leading-space open (still Open)
        ]
    }

    fn table_from(vocab: &[&str]) -> Vec<TokenDelim> {
        vocab.iter().map(|s| classify_str(s)).collect()
    }

    fn forced_from(vocab: &[&str]) -> ForcedIds {
        build_forced_ids(vocab.len(), |id| {
            vocab.get(id as usize).map(|s| s.to_string())
        })
    }

    #[test]
    fn classify_basics() {
        assert_eq!(classify_str("("), TokenDelim::Open(b'('));
        assert_eq!(classify_str(" ("), TokenDelim::Open(b'('));
        assert_eq!(classify_str(")"), TokenDelim::Close(b')'));
        assert_eq!(classify_str("\""), TokenDelim::Quote(b'"'));
        assert_eq!(classify_str("\"\"\""), TokenDelim::TripleQuote(b'"'));
        assert_eq!(classify_str(":"), TokenDelim::Colon);
        assert_eq!(classify_str("\n"), TokenDelim::Newline);
        assert_eq!(classify_str("    \n"), TokenDelim::Newline);
        assert_eq!(classify_str("foo"), TokenDelim::Other);
        assert_eq!(classify_str("  "), TokenDelim::Other);
        assert_eq!(classify_str("()"), TokenDelim::Opaque);
        assert_eq!(classify_str("):"), TokenDelim::Opaque);
        assert_eq!(classify_str("(x"), TokenDelim::Opaque);
        assert_eq!(classify_str("\");\n"), TokenDelim::Opaque);
    }

    #[test]
    fn open_bracket_does_not_overwrite_argument_tokens() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        // committed: foo (  -> arguments `bar`, `foo` must NOT be overwritten with ')'.
        let committed = vec![10u32, 0];
        let mut drafts = vec![11u32, 10, 1];
        let before = drafts.clone();
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(stats.splices, 0);
        assert_eq!(drafts, before);
    }

    #[test]
    fn splice_repairs_mismatched_close_bracket_after_arguments() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        // committed: foo (  -> draft [bar, '}', foo]: keep 'bar', repair '}' -> ')'.
        let committed = vec![10u32, 0];
        let mut drafts = vec![11u32, 5, 10];
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(stats.splices, 1);
        assert_eq!(stats.first_pos, 1);
        assert_eq!(drafts, vec![11, 1, 10]);
    }

    #[test]
    fn splice_noop_when_drafter_already_matches_closer() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        let committed = vec![10u32, 0];
        let mut drafts = vec![1u32, 10, 11];
        let before = drafts.clone();
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(stats.splices, 0);
        assert_eq!(drafts, before);
    }

    #[test]
    fn nested_brackets_repair_mismatched_closers_innermost_first() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        // committed: ( [  -> drafter proposes [bar, '}', '}']: repair '}' -> ']', then '}' -> ')'.
        let committed = vec![0u32, 2];
        let mut drafts = vec![11u32, 5, 5];
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(stats.splices, 2);
        assert_eq!(stats.first_pos, 1);
        assert_eq!(drafts, vec![11, 3, 1]);
    }

    #[test]
    fn string_suppresses_bracket_repair_and_never_forces_closing_quote() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        // committed: " (   -> inside double-quoted string; neither content nor '}' is spliced.
        let committed = vec![6u32, 0];
        let mut drafts = vec![10u32, 5, 11];
        let before = drafts.clone();
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(stats.splices, 0);
        assert_eq!(drafts, before);
    }

    #[test]
    fn opaque_token_in_committed_prefix_poisons_and_halts_splicing() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        // committed: foo ( then OPAQUE "(x" -> base poisons -> 0 splices even on mismatched '}'.
        let committed = vec![10u32, 0, 13];
        let mut drafts = vec![5u32, 10];
        let before = drafts.clone();
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(stats.splices, 0);
        assert_eq!(drafts, before);
    }

    #[test]
    fn triple_quote_does_not_overwrite_docstring_body() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        let committed = vec![12u32];
        let mut drafts = vec![10u32, 11, 5];
        let before = drafts.clone();
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(stats.splices, 0);
        assert_eq!(drafts, before);
    }

    #[test]
    fn no_forcing_when_balanced() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        let committed = vec![10u32, 0, 11, 1];
        let mut drafts = vec![10u32, 5, 10];
        let before = drafts.clone();
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(stats.splices, 0);
        assert_eq!(drafts, before);
    }

    #[test]
    fn drafts_len_preserved_on_multi_repair() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        let committed = vec![0u32, 2, 4]; // ( [ {
        let mut drafts = vec![1u32, 1, 3, 10, 11]; // wrong closers: ')' for '{', ')' for '[', ']' for '('
        let n = drafts.len();
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(drafts.len(), n);
        assert_eq!(stats.splices, 3);
        assert_eq!(drafts, vec![5, 3, 1, 10, 11]);
    }
}
