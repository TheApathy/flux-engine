// SPDX-License-Identifier: AGPL-3.0-only

//! Misc grammar compilation: structural-tag, JSON schema, JSON, EBNF.

use xgrammar::CompiledGrammar;

use super::engine::{GrammarEngine, GrammarError};

fn sanitize_json_schema_str(schema: &str) -> Result<String, GrammarError> {
    let parsed: serde_json::Value =
        serde_json::from_str(schema).map_err(|e| GrammarError::InvalidSchema(e.to_string()))?;
    let sanitized = super::schema::sanitize_schema_for_grammar(&parsed).ok_or_else(|| {
        GrammarError::InvalidSchema("JSON schema is unsatisfiable (`false`)".to_string())
    })?;
    Ok(sanitized.to_string())
}

impl GrammarEngine {
    /// Build and compile a structural tag grammar from raw JSON components,
    /// exposing `at_least_one` and `stop_after_first` parameters directly.
    pub(super) fn compile_structural_tag_raw(
        &mut self,
        triggers: &[String],
        tags: &[serde_json::Value],
        at_least_one: bool,
        stop_after_first: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        let structural_tag_json = serde_json::json!({
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": triggers,
                "tags": tags,
                "at_least_one": at_least_one,
                "stop_after_first": stop_after_first,
            }
        })
        .to_string();

        let grammar = xgrammar::Grammar::from_structural_tag(&structural_tag_json)
            .map_err(GrammarError::Compilation)?;
        self.compiler
            .compile_grammar(&grammar)
            .map_err(GrammarError::Compilation)
    }

    /// Build and compile a joint union grammar accepting EITHER one or more
    /// tool calls matching `tags` (with optional surrounding/inter-call
    /// whitespace) OR a JSON value matching `response_format` (either raw JSON
    /// or wrapped in an optional markdown `json` code fence via
    /// `with_optional_json_fence()`, forbidding arbitrary prose).
    ///
    /// Used when an OpenAI request specifies both `tools` and `response_format`
    /// under `tool_choice="auto"`: turn N can emit `<tool_call>...</tool_call>`
    /// and turn N+1 is constrained to schema-valid JSON (with an optional JSON
    /// code fence).
    pub(super) fn compile_structural_tag_or_response_format(
        &mut self,
        tags: &[serde_json::Value],
        stop_after_first: bool,
        response_format: &crate::api::ResponseFormatConstraint,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tags.is_empty() {
            return Err(GrammarError::NoTools);
        }
        let single_tag_format = if tags.len() == 1 {
            tags[0].clone()
        } else {
            serde_json::json!({
                "type": "or",
                "elements": tags,
            })
        };
        let single_tag_json = serde_json::json!({
            "type": "structural_tag",
            "format": single_tag_format,
        })
        .to_string();
        let single_tag = xgrammar::Grammar::from_structural_tag(&single_tag_json)
            .map_err(GrammarError::Compilation)?;
        let tool_branch = single_tag
            .one_or_more_ws_separated(stop_after_first)
            .map_err(GrammarError::Compilation)?;

        let rf_inner = match response_format {
            crate::api::ResponseFormatConstraint::JsonObject => {
                xgrammar::Grammar::builtin_json_grammar().map_err(GrammarError::Compilation)?
            }
            crate::api::ResponseFormatConstraint::JsonSchema { schema, strict } => {
                let sanitized = sanitize_json_schema_str(schema)?;
                xgrammar::Grammar::from_json_schema(
                    &sanitized,
                    true,
                    None,
                    None::<(&str, &str)>,
                    *strict,
                    None,
                    false,
                )
                .map_err(GrammarError::Compilation)?
            }
        };
        let rf_branch = rf_inner
            .with_optional_json_fence()
            .map_err(GrammarError::Compilation)?;

        let union = xgrammar::Grammar::union(&[&tool_branch, &rf_branch])
            .map_err(GrammarError::Compilation)?;
        self.compiler
            .compile_grammar(&union)
            .map_err(GrammarError::Compilation)
    }

    // ── JSON schema grammar ──

    /// Compile a grammar that enforces a JSON schema (strict mode enabled).
    pub fn compile_json_schema(&mut self, schema: &str) -> Result<CompiledGrammar, GrammarError> {
        self.compile_json_schema_with_strict(schema, true)
    }

    /// Compile a grammar that enforces a JSON schema with configurable
    /// XGrammar `strict_mode`.
    pub fn compile_json_schema_with_strict(
        &mut self,
        schema: &str,
        strict_mode: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        let sanitized = sanitize_json_schema_str(schema)?;
        self.compiler
            .compile_json_schema(
                &sanitized,
                true,                 // any_whitespace
                None,                 // indent
                None::<(&str, &str)>, // separators
                strict_mode,          // strict_mode
                None,                 // max_whitespace_cnt
            )
            .map_err(GrammarError::Compilation)
    }

    /// Compile the built-in JSON grammar (any valid JSON).
    pub fn compile_json_grammar(&mut self) -> Result<CompiledGrammar, GrammarError> {
        self.compiler
            .compile_builtin_json_grammar()
            .map_err(GrammarError::Compilation)
    }

    /// Compile a grammar from an EBNF string.
    pub fn compile_ebnf(
        &mut self,
        ebnf: &str,
        root_rule: &str,
    ) -> Result<CompiledGrammar, GrammarError> {
        self.compiler
            .compile_grammar_from_ebnf(ebnf, root_rule)
            .map_err(GrammarError::Compilation)
    }
}
