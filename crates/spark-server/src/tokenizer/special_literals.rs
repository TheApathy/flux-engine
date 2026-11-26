// SPDX-License-Identifier: AGPL-3.0-only

//! Request text cannot inject special tokens into a rendered chat template.
//!
//! The chat path renders the Jinja template into one string and tokenizes it.
//! The tokenizer matches added tokens anywhere in that string, so a message
//! containing `<|im_end|><|im_start|>system` or `<|image_pad|>` literally
//! became those control tokens. Here every special-token literal inside the
//! request's own strings (message content, tool calls, tool definitions) is
//! swapped for an opaque placeholder before rendering. Afterwards the special
//! tokens that remain are the template's own markers; the text between them is
//! encoded with the placeholders restored and special-token matching off, so
//! request literals tokenize as plain text (HF `split_special_tokens=True`
//! for that span). Non-special added tokens (`<think>`, `<tool_call>`,
//! `</tool_response>`) are not escaped in any role, tool and user content
//! included: Qwen templates branch on them in content (`'</think>' in
//! content` and the split on it for assistant turns, and
//! `content.startswith('<tool_response>')` on user turns to find the last
//! real query), so placeholders there would change which template branch
//! runs. Those tokens are ordinary vocabulary to the model; the special ones
//! above are what delimit turns and media.

use std::borrow::Cow;
use std::sync::OnceLock;

use aho_corasick::{AhoCorasick, MatchKind};
use anyhow::Result;
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

/// Brackets a placeholder index. Supplementary private-use code points that
/// no chat template emits; request text that already contains one is itself
/// swapped for a placeholder, so every bracket after rendering is ours.
const OPEN: char = '\u{F0000}';
const CLOSE: char = '\u{F0001}';

pub(super) struct SpecialLiterals {
    /// Id of each special added token, in `markers` pattern order.
    ids: Vec<u32>,
    /// Special-token contents, leftmost-longest: the template markers.
    markers: Option<AhoCorasick>,
    /// Those contents plus the placeholder brackets: what request text escapes.
    escapes: Option<AhoCorasick>,
    /// The same tokenizer with special-token matching off; built on first use.
    literal: OnceLock<Tokenizer>,
}

/// Request strings with special literals swapped out, and the originals.
pub(crate) struct Escaped<'a> {
    pub(crate) messages: Cow<'a, [Value]>,
    pub(super) tools: Option<Cow<'a, [Value]>>,
    originals: Vec<String>,
}

impl<'a> Escaped<'a> {
    /// The request as-is (`--allow-special-tokens-in-content`).
    pub(super) fn unescaped(messages: &'a [Value], tools: Option<&'a [Value]>) -> Self {
        Self {
            messages: Cow::Borrowed(messages),
            tools: tools.map(Cow::Borrowed),
            originals: Vec::new(),
        }
    }

    /// How many literals were kept as text.
    pub(crate) fn count(&self) -> usize {
        self.originals.len()
    }
}

impl SpecialLiterals {
    pub(super) fn from_tokenizer(tokenizer: &Tokenizer) -> Self {
        let mut specials: Vec<(String, u32)> = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, token)| token.special && !token.content.is_empty())
            .map(|(id, token)| (token.content, id))
            .collect();
        specials.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let build = |patterns: &[String]| {
            (!patterns.is_empty()).then(|| {
                AhoCorasick::builder()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build(patterns)
                    .expect("special-token automaton")
            })
        };
        let contents: Vec<String> = specials
            .iter()
            .map(|(content, _)| content.clone())
            .collect();
        let mut escapes = contents.clone();
        if !contents.is_empty() {
            escapes.extend([OPEN.to_string(), CLOSE.to_string()]);
        }
        Self {
            ids: specials.iter().map(|&(_, id)| id).collect(),
            markers: build(&contents),
            escapes: build(&escapes),
            literal: OnceLock::new(),
        }
    }

    /// Swap special literals (and stray placeholder brackets) in every string
    /// of `messages` and `tools`. Borrows unchanged input.
    pub(super) fn escape<'a>(
        &self,
        messages: &'a [Value],
        tools: Option<&'a [Value]>,
    ) -> Escaped<'a> {
        let mut originals = Vec::new();
        let messages = self.escape_values(messages, &mut originals);
        let tools = tools.map(|tools| self.escape_values(tools, &mut originals));
        Escaped {
            messages,
            tools,
            originals,
        }
    }

    fn escape_values<'a>(
        &self,
        values: &'a [Value],
        originals: &mut Vec<String>,
    ) -> Cow<'a, [Value]> {
        if !values.iter().any(|value| self.needs_escape(value)) {
            return Cow::Borrowed(values);
        }
        Cow::Owned(
            values
                .iter()
                .map(|value| {
                    let mut value = value.clone();
                    self.escape_value(&mut value, originals);
                    value
                })
                .collect(),
        )
    }

    fn needs_escape(&self, value: &Value) -> bool {
        match value {
            Value::String(text) => self.next_match(text, 0).is_some(),
            Value::Array(items) => items.iter().any(|item| self.needs_escape(item)),
            Value::Object(map) => map
                .iter()
                .any(|(key, item)| self.next_match(key, 0).is_some() || self.needs_escape(item)),
            _ => false,
        }
    }

    fn escape_value(&self, value: &mut Value, originals: &mut Vec<String>) {
        match value {
            Value::String(text) => {
                if let Cow::Owned(escaped) = self.escape_text(text, originals) {
                    *text = escaped;
                }
            }
            Value::Array(items) => items
                .iter_mut()
                .for_each(|item| self.escape_value(item, originals)),
            Value::Object(map) => {
                // Keys render too (tool-call argument names, schema property
                // names), so they are escaped like values.
                let entries = std::mem::take(map);
                *map = entries
                    .into_iter()
                    .map(|(key, mut item)| {
                        self.escape_value(&mut item, originals);
                        (self.escape_text(&key, originals).into_owned(), item)
                    })
                    .collect::<Map<String, Value>>();
            }
            _ => {}
        }
    }

    fn escape_text<'t>(&self, text: &'t str, originals: &mut Vec<String>) -> Cow<'t, str> {
        let Some(mut found) = self.next_match(text, 0) else {
            return Cow::Borrowed(text);
        };
        let mut out = String::with_capacity(text.len() + 16);
        let mut at = 0;
        loop {
            let (start, len) = found;
            out.push_str(&text[at..start]);
            out.push(OPEN);
            out.push_str(&originals.len().to_string());
            out.push(CLOSE);
            originals.push(text[start..start + len].to_string());
            at = start + len;
            match self.next_match(text, at) {
                Some(next) => found = next,
                None => break,
            }
        }
        out.push_str(&text[at..]);
        Cow::Owned(out)
    }

    /// First `(byte start, byte len)` at or after `from` of a special literal
    /// or a placeholder bracket character (one leftmost-longest scan).
    fn next_match(&self, text: &str, from: usize) -> Option<(usize, usize)> {
        let found = self.escapes.as_ref()?.find(&text[from..])?;
        Some((from + found.start(), found.len()))
    }

    /// Encode a template rendered from [`Escaped`] input. With nothing
    /// escaped this is equivalent to `tokenizer.encode_fast(rendered, false)`
    /// (or `tokenizer.encode(rendered, false)` when `ids_only` is disabled),
    /// gated by `gate` for `>= 1 MiB` inputs.
    pub(super) fn encode(
        &self,
        tokenizer: &Tokenizer,
        rendered: &str,
        escaped: &Escaped<'_>,
        ids_only: bool,
        gate: &super::large_encode::Gate,
    ) -> Result<Vec<u32>> {
        if escaped.originals.is_empty() {
            return encode_with(tokenizer, rendered, ids_only, Some(gate));
        }
        // Acquire at most one permit across the entire rendered prompt so a
        // large escaped prompt neither bypasses the gate when split into <1 MiB
        // segments nor self-deadlocks by double-acquiring inside segments.
        let _permit = gate.acquire(rendered.len());
        let literal = self.literal.get_or_init(|| {
            let mut literal = tokenizer.clone();
            literal.set_encode_special_tokens(true);
            literal
        });
        let markers = self
            .markers
            .as_ref()
            .expect("escaping implies special tokens");
        let mut ids = Vec::new();
        let mut segment_start = 0;
        for marker in markers.find_iter(rendered) {
            self.encode_segment(
                literal,
                &rendered[segment_start..marker.start()],
                escaped,
                ids_only,
                &mut ids,
            )?;
            ids.push(self.ids[marker.pattern().as_usize()]);
            segment_start = marker.end();
        }
        self.encode_segment(
            literal,
            &rendered[segment_start..],
            escaped,
            ids_only,
            &mut ids,
        )?;
        Ok(ids)
    }

    fn encode_segment(
        &self,
        literal: &Tokenizer,
        segment: &str,
        escaped: &Escaped<'_>,
        ids_only: bool,
        ids: &mut Vec<u32>,
    ) -> Result<()> {
        if segment.is_empty() {
            return Ok(());
        }
        let restored = restore(segment, &escaped.originals)?;
        ids.extend(encode_with(literal, &restored, ids_only, None)?);
        Ok(())
    }
}

fn encode_with(
    tokenizer: &Tokenizer,
    text: &str,
    ids_only: bool,
    gate: Option<&super::large_encode::Gate>,
) -> Result<Vec<u32>> {
    let _permit = gate.and_then(|g| g.acquire(text.len()));
    let encoding = super::ids_only::encode(tokenizer, text, ids_only)
        .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {e}"))?;
    Ok(encoding.get_ids().to_vec())
}

/// Put the original literals back. A bracket the template split or dropped
/// cannot be restored faithfully, so it is an error, never a guess.
fn restore(segment: &str, originals: &[String]) -> Result<String> {
    let mut out = String::with_capacity(segment.len());
    let mut rest = segment;
    while let Some(open) = rest.find(OPEN) {
        out.push_str(&rest[..open]);
        let after = &rest[open + OPEN.len_utf8()..];
        let close = after.find(CLOSE).ok_or_else(|| {
            anyhow::anyhow!("chat template split an escaped special-token literal")
        })?;
        let original = after[..close]
            .parse::<usize>()
            .ok()
            .and_then(|index| originals.get(index))
            .ok_or_else(|| {
                anyhow::anyhow!("chat template altered an escaped special-token literal")
            })?;
        out.push_str(original);
        rest = &after[close + CLOSE.len_utf8()..];
    }
    anyhow::ensure!(
        !rest.contains(CLOSE),
        "chat template split an escaped special-token literal"
    );
    out.push_str(rest);
    Ok(out)
}
