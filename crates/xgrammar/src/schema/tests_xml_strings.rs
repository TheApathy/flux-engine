// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::grammar::parse_ebnf_default;

#[test]
fn qwen_xml_nonempty_string_keeps_the_close_tag_exclusion() {
    let got = qwen_xml_tool_calling_to_ebnf(
        r#"{"type":"object","properties":{"loc":{"type":"string","minLength":1},"unit":{"type":"string"}},"required":["loc"]}"#,
    )
    .unwrap();
    assert!(!got.contains("[^]{1,}"), "{got}");
    assert!(got.contains("xml_string_nonempty ::= "), "{got}");
    assert!(got.contains("xml_string_nonempty_q11 ::= "), "{got}");
    assert!(got.contains("\"</parameter>\" root_part_0"), "{got}");
    assert!(got.contains("[ \\n\\t]* \"<parameter=unit>\""), "{got}");
    assert!(parse_ebnf_default(&got).is_ok(), "{got}");
}

#[test]
fn non_qwen_xml_retains_existing_required_string_rules() {
    let schema = r#"{"type":"object","properties":{"loc":{"type":"string","minLength":1}},"required":["loc"]}"#;
    for grammar in [
        minimax_xml_tool_calling_to_ebnf(schema).unwrap(),
        deepseek_xml_tool_calling_to_ebnf(schema).unwrap(),
    ] {
        assert!(!grammar.contains("xml_string_nonempty"), "{grammar}");
        assert!(grammar.contains("[^]{1,}"), "{grammar}");
        assert!(parse_ebnf_default(&grammar).is_ok(), "{grammar}");
    }
}
