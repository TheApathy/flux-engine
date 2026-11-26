// SPDX-License-Identifier: AGPL-3.0-only
//
// Scalar-type EBNF generators — port of
// `JSONSchemaConverter::Generate{Integer,Number,String,Any,Const,Enum}`
// and the XML overrides from `cpp/json_schema_converter*.cc`.

use super::converter::{
    BASIC_ARRAY, BASIC_BOOLEAN, BASIC_NULL, BASIC_NUMBER, BASIC_OBJECT, BASIC_STRING,
    BASIC_STRING_SUB, JsonSchemaConverter, XML_OBJECT, XML_STRING, XML_STRING_NONEMPTY,
};
use super::error::{SchemaError, SchemaResult};
use super::float_regex::generate_float_range_regex;
use super::formats::format_to_regex;
use super::options::JsonFormat;
use super::range_regex::generate_range_regex;
use super::spec::{IntegerSpec, NumberSpec, StringSpec};
use crate::regex::regex_to_ebnf;

/// Convert a regex to an EBNF body, mapping converter errors into a
/// [`SchemaError`]. `with_rule_name = false` matches the C++
/// `RegexToEBNF(..., false)` calls.
fn regex_body(regex: &str) -> SchemaResult<String> {
    regex_to_ebnf(regex, false)
        .map_err(|e| SchemaError::invalid(format!("regex conversion failed: {e}")))
}

impl<'p> JsonSchemaConverter<'p> {
    /// Generate the body for an integer spec. Port of `GenerateInteger`.
    pub(super) fn generate_integer(&self, spec: &IntegerSpec) -> String {
        let mut start = spec.minimum;
        let mut end = spec.maximum;
        if let Some(em) = spec.exclusive_minimum {
            let ex = em.saturating_add(1);
            start = Some(start.map_or(ex, |s| s.max(ex)));
        }
        if let Some(em) = spec.exclusive_maximum {
            let ex = em.saturating_sub(1);
            end = Some(end.map_or(ex, |e| e.min(ex)));
        }
        if start.is_some() || end.is_some() {
            let regex = generate_range_regex(start, end);
            // Integer range regex is always well-formed.
            return regex_body(&regex).unwrap_or_default();
        }
        "(\"0\" | \"-\"? [1-9] [0-9]*)".to_string()
    }

    /// Generate the body for a number spec. Port of `GenerateNumber`.
    pub(super) fn generate_number(&self, spec: &NumberSpec) -> String {
        let mut start = spec.minimum;
        let mut end = spec.maximum;
        if let Some(em) = spec.exclusive_minimum {
            start = Some(start.map_or(em, |s| s.max(em)));
        }
        if let Some(em) = spec.exclusive_maximum {
            end = Some(end.map_or(em, |e| e.min(em)));
        }
        if start.is_some() || end.is_some() {
            let regex = generate_float_range_regex(start, end, 6);
            return regex_body(&regex).unwrap_or_default();
        }
        "\"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?".to_string()
    }

    /// Generate the body for a string spec. Port of `GenerateString`
    /// plus the `XMLToolCallingConverter::GenerateString` override.
    pub(super) fn generate_string(&self, spec: &StringSpec) -> SchemaResult<String> {
        if self.at_xml_layer() {
            // XML layer: bare strings dispatch via TagDispatch.
            if spec.pattern.is_none()
                && spec.format.is_none()
                && spec.min_length == 0
                && spec.max_length == -1
            {
                return Ok(XML_STRING.to_string());
            }
            if let Some(fmt) = &spec.format
                && let Some(regex) = format_to_regex(fmt)
            {
                return regex_body(&regex);
            }
            if let Some(pattern) = &spec.pattern {
                return regex_body(pattern);
            }
            // Required strings gain minLength 1 during tool-schema sanitization.
            // An any-byte repetition also consumed the parameter's close tag.
            if spec.min_length == 1 && spec.max_length == -1 && self.format == JsonFormat::QwenXml {
                return Ok(XML_STRING_NONEMPTY.to_string());
            }
            // Other length bounds retain the upstream form and its existing
            // close-tag limitation. Pattern/format handling also stays unchanged.
            if spec.min_length != 0 || spec.max_length != -1 {
                let repetition = if spec.max_length == -1 {
                    format!("{{{},}}", spec.min_length)
                } else {
                    format!("{{{},{}}}", spec.min_length, spec.max_length)
                };
                return Ok(format!("[^]{repetition}"));
            }
        }

        // JSON-style string.
        if let Some(fmt) = &spec.format
            && let Some(regex) = format_to_regex(fmt)
        {
            let converted = regex_body(&regex)?;
            return Ok(format!("\"\\\"\" {converted} \"\\\"\""));
        }
        if let Some(pattern) = &spec.pattern {
            let converted = regex_body(pattern)?;
            return Ok(format!("\"\\\"\" {converted} \"\\\"\""));
        }
        if spec.min_length != 0 || spec.max_length != -1 {
            let char_pattern = "[^\"\\\\\\r\\n]";
            let repetition = if spec.max_length == -1 {
                format!("{{{},}}", spec.min_length)
            } else {
                format!("{{{},{}}}", spec.min_length, spec.max_length)
            };
            return Ok(format!("\"\\\"\" {char_pattern}{repetition} \"\\\"\""));
        }
        Ok(format!("[\"] {BASIC_STRING_SUB}"))
    }

    /// Generate the body for the "any" spec. Port of `GenerateAny`
    /// plus the XML override (upstream commit 41dbbb1, #634).
    pub(super) fn generate_any(&self, _rule_name: &str) -> String {
        if self.is_xml() {
            // Root XML layer: an arbitrary value is an XML object.
            if self.nested_object_level == 0 {
                return XML_OBJECT.to_string();
            }
            // XML param layer: a string, array, or object.
            if self.nested_object_level == 1 {
                return format!("{XML_STRING} | {BASIC_ARRAY} | {BASIC_OBJECT}");
            }
        }
        format!(
            "{BASIC_NUMBER} | {BASIC_STRING} | {BASIC_BOOLEAN} | {BASIC_NULL} | \
             {BASIC_ARRAY} | {BASIC_OBJECT}"
        )
    }

    /// Generate the body for a `const`. `json_value` is serialized
    /// JSON. Port of `GenerateConst` plus the XML override.
    pub(super) fn generate_const(&self, json_value: &str) -> String {
        if self.at_xml_layer() {
            if json_value.len() >= 2 && json_value.starts_with('"') && json_value.ends_with('"') {
                return format!("\"{}\"", &json_value[1..json_value.len() - 1]);
            }
            return format!("\"{}\"", json_str_to_printable(json_value));
        }
        format!("\"{}\"", json_str_to_printable(json_value))
    }

    /// Generate the body for an `enum`. Port of `GenerateEnum` plus
    /// the XML override.
    pub(super) fn generate_enum(&self, values: &[String]) -> String {
        let mut out = String::new();
        for (i, val) in values.iter().enumerate() {
            if i != 0 {
                out.push_str(" | ");
            }
            if self.at_xml_layer() {
                if val.len() >= 2 && val.starts_with('"') && val.ends_with('"') {
                    out.push_str(&format!("(\"{}\")", &val[1..val.len() - 1]));
                } else {
                    out.push_str(&format!("(\"{}\")", json_str_to_printable(val)));
                }
            } else {
                out.push_str(&format!("(\"{}\")", json_str_to_printable(val)));
            }
        }
        out
    }
}

/// Escape `"` and `\` in a JSON-serialized value so it can be placed
/// inside an EBNF double-quoted literal. Port of `JSONStrToPrintableStr`.
pub(super) fn json_str_to_printable(json_str: &str) -> String {
    json_str.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::json_str_to_printable;
    use crate::earley::EarleyParser;
    use crate::grammar::functor::{GrammarNormalizer, GrammarOptimizer};
    use crate::grammar::parse_ebnf_default;
    use crate::schema::{
        SchemaConverterOptions, json_schema_to_ebnf, json_schema_to_grammar,
        qwen_xml_tool_calling_to_ebnf,
    };

    fn accepts(schema: &str, input: &str) -> bool {
        let opts = SchemaConverterOptions {
            any_whitespace: false,
            ..SchemaConverterOptions::default()
        };
        let g = json_schema_to_grammar(schema, &opts).expect("schema should compile");
        let opt = Arc::new(GrammarOptimizer::apply(GrammarNormalizer::apply(g)));
        let mut p = EarleyParser::from_grammar(opt);
        for b in input.bytes() {
            if !p.advance(b) {
                return false;
            }
        }
        p.is_completed()
    }

    #[test]
    fn printable_escapes_quotes_and_backslash() {
        assert_eq!(json_str_to_printable(r#""ab""#), r#"\"ab\""#);
        assert_eq!(json_str_to_printable(r"a\b"), r"a\\b");
    }

    #[test]
    fn exclusive_bounds_do_not_loosen_tighter_inclusive_bounds() {
        let int_schema = r#"{"type":"integer","minimum":10,"exclusiveMinimum":0,"maximum":20,"exclusiveMaximum":100}"#;
        assert!(accepts(int_schema, "10"));
        assert!(accepts(int_schema, "20"));
        assert!(!accepts(int_schema, "5"));
        assert!(!accepts(int_schema, "50"));

        let num_schema = r#"{"type":"number","minimum":10.0,"exclusiveMinimum":0.0,"maximum":20.0,"exclusiveMaximum":100.0}"#;
        assert!(accepts(num_schema, "10.0"));
        assert!(accepts(num_schema, "15.5"));
        assert!(accepts(num_schema, "20.0"));
        assert!(!accepts(num_schema, "5.0"));
        assert!(!accepts(num_schema, "50.0"));
    }

    #[test]
    fn xml_const_and_enum_non_string_values_escape_quotes() {
        let const_schema =
            r#"{"type":"object","properties":{"cfg":{"const":{"a":"b"}}},"required":["cfg"]}"#;
        let const_ebnf = qwen_xml_tool_calling_to_ebnf(const_schema).unwrap();
        assert!(parse_ebnf_default(&const_ebnf).is_ok(), "{const_ebnf}");

        let enum_schema = r#"{"type":"object","properties":{"cfg":{"enum":[{"k":"v"},["x","y"]]}},"required":["cfg"]}"#;
        let enum_ebnf = qwen_xml_tool_calling_to_ebnf(enum_schema).unwrap();
        assert!(parse_ebnf_default(&enum_ebnf).is_ok(), "{enum_ebnf}");

        // JSON mode remains parseable too.
        let opts = SchemaConverterOptions::default();
        let json_ebnf = json_schema_to_ebnf(enum_schema, &opts).unwrap();
        assert!(parse_ebnf_default(&json_ebnf).is_ok(), "{json_ebnf}");
    }
}
