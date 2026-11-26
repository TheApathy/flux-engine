// SPDX-License-Identifier: AGPL-3.0-only
//
// `Grammar` handle — W7 compatibility shim.
//
// The vendored C++ `xgrammar-rs` exposed a `Grammar` handle type
// (`from_ebnf`, `from_json_schema`, `from_regex`, `from_structural_tag`,
// `builtin_json_grammar`) which `GrammarCompiler::compile_grammar`
// consumed by `&Grammar`. The pure-Rust port works directly on the
// `GrammarData` AST and has no separate handle type.
//
// This thin newtype restores the vendored surface: it owns a parsed,
// **un-normalized** `GrammarData` (the compiler runs normalization).
// Atlas's `grammar/compile_misc.rs` uses `Grammar::from_structural_tag`
// followed by `compiler.compile_grammar(&grammar)`.

use crate::grammar::functor::{GrammarNormalizer, GrammarUnion, add_sub_grammar};
use crate::grammar::{GrammarBuilder, GrammarData, parse_ebnf};
use crate::schema::{SchemaConverterOptions, builtin_json_grammar_ebnf, json_schema_to_ebnf};
use crate::structural_tag::structural_tag_to_grammar;

/// A parsed grammar, ready to be handed to
/// [`crate::GrammarCompiler::compile_grammar`].
///
/// Port of the vendored `xgrammar::Grammar` handle.
#[derive(Debug, Clone)]
pub struct Grammar {
    data: GrammarData,
}

impl Grammar {
    /// Construct a grammar from an EBNF string.
    ///
    /// Port of the vendored `Grammar::from_ebnf`.
    pub fn from_ebnf(ebnf_string: &str, root_rule_name: &str) -> Result<Self, String> {
        parse_ebnf(ebnf_string, root_rule_name)
            .map(|data| Self { data })
            .map_err(|e| e.to_string())
    }

    /// Construct a grammar from a JSON schema.
    ///
    /// Port of the vendored `Grammar::from_json_schema`. `indent`,
    /// `separators` and `max_whitespace_cnt` follow the same convention
    /// as `json.dumps()`; `print_converted_ebnf` is accepted for parity
    /// and has no effect.
    pub fn from_json_schema(
        schema: &str,
        any_whitespace: bool,
        indent: Option<i32>,
        separators: Option<(impl AsRef<str>, impl AsRef<str>)>,
        strict_mode: bool,
        max_whitespace_cnt: Option<i32>,
        print_converted_ebnf: bool,
    ) -> Result<Self, String> {
        let _ = print_converted_ebnf;
        let options = SchemaConverterOptions {
            any_whitespace,
            indent,
            separators: separators.map(|(c, s)| (c.as_ref().to_string(), s.as_ref().to_string())),
            strict_mode,
            max_whitespace_cnt,
            ..SchemaConverterOptions::default()
        };
        let ebnf = json_schema_to_ebnf(schema, &options).map_err(|e| e.to_string())?;
        Self::from_ebnf(&ebnf, "root")
    }

    /// Construct a grammar from a structural tag JSON document.
    ///
    /// Port of the vendored `Grammar::from_structural_tag`.
    pub fn from_structural_tag(structural_tag_json: &str) -> Result<Self, String> {
        structural_tag_to_grammar(structural_tag_json)
            .map(|data| Self { data })
            .map_err(|e| e.to_string())
    }

    /// Build the standard JSON grammar.
    ///
    /// Port of the vendored `Grammar::builtin_json_grammar`.
    pub fn builtin_json_grammar() -> Result<Self, String> {
        Self::from_ebnf(&builtin_json_grammar_ebnf(), "root")
    }

    /// Construct the union (`g_0 | g_1 | ...`) of one or more grammars.
    ///
    /// Port of upstream `Grammar::Union`.
    pub fn union(grammars: &[&Grammar]) -> Result<Self, String> {
        if grammars.is_empty() {
            return Err("Grammar::union requires at least one grammar".to_string());
        }
        if grammars.len() == 1 {
            return Ok(grammars[0].clone());
        }
        let grammar_data: Vec<GrammarData> = grammars.iter().map(|g| g.data.clone()).collect();
        let combined = GrammarNormalizer::apply(GrammarUnion::apply(&grammar_data));
        Ok(Self { data: combined })
    }

    /// Wrap this grammar with optional ASCII whitespace (`[ \t\r\n]*`) on both
    /// sides so leading newlines (e.g. immediately after `</think>`) and
    /// trailing newlines are accepted without weakening inner structure.
    pub fn with_optional_ws(&self) -> Result<Self, String> {
        let ws = Self::from_ebnf("root ::= [ \\t\\r\\n]*\n", "root")?;
        let mut builder = GrammarBuilder::new();
        let root_id = builder.add_empty_rule("root").map_err(|e| e.to_string())?;
        let ws_id = add_sub_grammar(&mut builder, &ws.data);
        let item_id = add_sub_grammar(&mut builder, &self.data);
        let r_ws_pre = builder.add_rule_ref(ws_id);
        let r_item = builder.add_rule_ref(item_id);
        let r_ws_post = builder.add_rule_ref(ws_id);
        let seq = builder.add_sequence(&[r_ws_pre, r_item, r_ws_post]);
        let body = builder.add_choices(&[seq]);
        builder
            .update_rule_body(root_id, body)
            .map_err(|e| e.to_string())?;
        let data = builder.get_by_id(root_id).map_err(|e| e.to_string())?;
        Ok(Self {
            data: GrammarNormalizer::apply(data),
        })
    }

    /// Wrap a JSON grammar so it accepts EITHER raw JSON surrounded by
    /// optional ASCII whitespace OR markdown-fenced JSON
    /// (`` ```json\n<self>\n``` `` / `` ```\n<self>\n``` ``) while still
    /// strictly enforcing `self` on every byte of the JSON payload and
    /// forbidding any leading or trailing prose.
    ///
    /// Used in joint `tools | response_format` unions where thinking models
    /// strongly prefer opening nested JSON with `` ``` `` after `</think>`:
    /// masking `` ``` `` while leaving `<tool_call>` legal causes greedy
    /// decoding to collapse into spurious tool calls instead of emitting JSON.
    pub fn with_optional_json_fence(&self) -> Result<Self, String> {
        let ws = Self::from_ebnf("root ::= [ \\t\\r\\n]*\n", "root")?;
        let fence_open = Self::from_ebnf(
            "root ::= [ \\t\\r\\n]* \"```\" (\"json\" | \"JSON\")? [ \\t]* \"\\r\"? \"\\n\" [ \\t\\r\\n]*\n",
            "root",
        )?;
        let fence_close =
            Self::from_ebnf("root ::= [ \\t\\r\\n]* \"```\" [ \\t\\r\\n]*\n", "root")?;
        let mut builder = GrammarBuilder::new();
        let root_id = builder.add_empty_rule("root").map_err(|e| e.to_string())?;
        let ws_id = add_sub_grammar(&mut builder, &ws.data);
        let open_id = add_sub_grammar(&mut builder, &fence_open.data);
        let close_id = add_sub_grammar(&mut builder, &fence_close.data);
        let item_id = add_sub_grammar(&mut builder, &self.data);

        // Branch 1: raw JSON with optional surrounding ASCII whitespace.
        let raw_pre = builder.add_rule_ref(ws_id);
        let raw_item = builder.add_rule_ref(item_id);
        let raw_post = builder.add_rule_ref(ws_id);
        let raw_seq = builder.add_sequence(&[raw_pre, raw_item, raw_post]);

        // Branch 2: markdown-fenced JSON (` ```json\n<self>\n``` `), no prose.
        let fenced_open = builder.add_rule_ref(open_id);
        let fenced_item = builder.add_rule_ref(item_id);
        let fenced_close = builder.add_rule_ref(close_id);
        let fenced_seq = builder.add_sequence(&[fenced_open, fenced_item, fenced_close]);

        let body = builder.add_choices(&[raw_seq, fenced_seq]);
        builder
            .update_rule_body(root_id, body)
            .map_err(|e| e.to_string())?;
        let data = builder.get_by_id(root_id).map_err(|e| e.to_string())?;
        Ok(Self {
            data: GrammarNormalizer::apply(data),
        })
    }

    /// Match one or more occurrences of `self` separated (and surrounded) by
    /// optional ASCII whitespace (`[ \t\r\n]*`).
    ///
    /// When `stop_after_first` is `true`, matches exactly one occurrence
    /// (`ws self ws`); when `false`, matches one or more (`ws self (ws self)* ws`).
    pub fn one_or_more_ws_separated(&self, stop_after_first: bool) -> Result<Self, String> {
        if stop_after_first {
            return self.with_optional_ws();
        }
        let ws = Self::from_ebnf("root ::= [ \\t\\r\\n]*\n", "root")?;
        let mut builder = GrammarBuilder::new();
        let root_id = builder.add_empty_rule("root").map_err(|e| e.to_string())?;
        let tail_id = builder
            .add_empty_rule_with_hint("ws_repeat_tail")
            .map_err(|e| e.to_string())?;
        let ws_id = add_sub_grammar(&mut builder, &ws.data);
        let item_id = add_sub_grammar(&mut builder, &self.data);

        // tail ::= ws item tail | ws
        let t_ws = builder.add_rule_ref(ws_id);
        let t_item = builder.add_rule_ref(item_id);
        let t_rec = builder.add_rule_ref(tail_id);
        let t_more = builder.add_sequence(&[t_ws, t_item, t_rec]);
        let t_done_ws = builder.add_rule_ref(ws_id);
        let t_done = builder.add_sequence(&[t_done_ws]);
        let tail_body = builder.add_choices(&[t_more, t_done]);
        builder
            .update_rule_body(tail_id, tail_body)
            .map_err(|e| e.to_string())?;

        // root ::= ws item tail
        let r_ws = builder.add_rule_ref(ws_id);
        let r_item = builder.add_rule_ref(item_id);
        let r_tail = builder.add_rule_ref(tail_id);
        let root_seq = builder.add_sequence(&[r_ws, r_item, r_tail]);
        let root_body = builder.add_choices(&[root_seq]);
        builder
            .update_rule_body(root_id, root_body)
            .map_err(|e| e.to_string())?;

        let data = builder.get_by_id(root_id).map_err(|e| e.to_string())?;
        Ok(Self {
            data: GrammarNormalizer::apply(data),
        })
    }

    /// The grammar in EBNF form.
    ///
    /// Port of the vendored `Grammar::to_string_ebnf`.
    pub fn to_string_ebnf(&self) -> String {
        crate::grammar::print_grammar(&self.data)
    }

    /// Borrow the underlying parsed AST.
    pub fn data(&self) -> &GrammarData {
        &self.data
    }

    /// Consume the handle, yielding the underlying AST.
    pub fn into_data(self) -> GrammarData {
        self.data
    }
}

impl core::fmt::Display for Grammar {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.to_string_ebnf())
    }
}

impl From<GrammarData> for Grammar {
    fn from(data: GrammarData) -> Self {
        Self { data }
    }
}
