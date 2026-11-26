// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::super::*;

fn tool(name: &str, required: &str) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: name.into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {required: {"type": "string"}},
                "required": [required]
            })),
        },
    }
}

fn call(name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        id: "call_0".into(),
        call_type: "function".into(),
        function: FunctionCall {
            name: name.into(),
            arguments: arguments.into(),
        },
    }
}

#[test]
fn missing_required_value_is_not_fabricated() {
    let weather = tool("get_weather", "city");
    let mut calls = vec![call("get_weather", "{}")];
    backfill_required_params(&mut calls, std::slice::from_ref(&weather));

    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert!(
        args.get("city").is_none(),
        "Atlas must not author `city`: {args}"
    );
    let issue = assess_tool_call(&calls[0], std::slice::from_ref(&weather)).unwrap_err();
    assert!(matches!(issue, ToolCallIssue::MissingParam(_)));
}

#[test]
fn model_authored_empty_string_is_preserved() {
    let weather = tool("get_weather", "city");
    let mut calls = vec![call("get_weather", r#"{"city":""}"#)];
    backfill_required_params(&mut calls, std::slice::from_ref(&weather));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "");
}

#[test]
fn derivable_description_is_still_filled() {
    let bash = ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "bash".into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["command", "description"]
            })),
        },
    };
    let mut calls = vec![call("bash", r#"{"command":"ls /tmp"}"#)];
    assert_eq!(
        backfill_required_params(&mut calls, std::slice::from_ref(&bash)),
        1
    );
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["description"], "Run: ls /tmp");
}

#[test]
fn repair_counts_only_actual_argument_changes() {
    let typed = ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "inspect".into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "file_path": {"type": "string"}
                }
            })),
        },
    };
    let mut calls = vec![call(
        "inspect",
        r#"{"count":"2","file_path":"/work/tree/file.txt"}"#,
    )];

    assert_eq!(
        backfill_required_params(&mut calls, std::slice::from_ref(&typed)),
        1
    );
    assert_eq!(
        backfill_required_params(&mut calls, std::slice::from_ref(&typed)),
        0,
        "an already repaired call must not be reported again"
    );
    assert_eq!(normalize_paths(&mut calls, "/work/tree"), 1);
    assert_eq!(normalize_paths(&mut calls, "/work/tree"), 0);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["count"], 2);
    assert_eq!(args["file_path"], "file.txt");
}

#[test]
fn ordinary_missing_param_is_attached_with_feedback() {
    let weather = tool("get_weather", "city");
    let result = validate_tool_calls(
        vec![call("get_weather", "{}")],
        std::slice::from_ref(&weather),
    );
    assert_eq!(
        result.valid.len(),
        1,
        "ordinary call remains client-visible"
    );
    assert_eq!(result.valid[0].function.arguments, "{}");
    assert_eq!(result.errors.len(), 1);
    assert!(result.errors[0].contains("city"));
}

#[test]
fn omitted_write_path_is_withheld() {
    let write = tool("write", "filePath");
    let result = validate_tool_calls(vec![call("write", "{}")], std::slice::from_ref(&write));
    assert!(result.valid.is_empty());
    assert_eq!(result.errors.len(), 1);
    assert!(result.errors[0].contains("filePath"));
}

#[test]
fn omitted_shell_command_is_withheld() {
    let exec = tool("exec", "command");
    let result = validate_tool_calls(vec![call("exec", "{}")], std::slice::from_ref(&exec));
    assert!(result.valid.is_empty());
    assert_eq!(result.errors.len(), 1);
    assert!(result.errors[0].contains("command"));
}

#[test]
fn critical_omission_wins_regardless_of_required_array_order() {
    for (name, first, critical) in [
        ("write", "content", "filePath"),
        ("exec", "description", "command"),
    ] {
        let definition = ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: name.into(),
                description: None,
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        first: {"type": "string"},
                        critical: {"type": "string"}
                    },
                    "required": [first, critical]
                })),
            },
        };
        let result = validate_tool_calls(vec![call(name, "{}")], std::slice::from_ref(&definition));
        assert!(result.valid.is_empty(), "{name} must be withheld");
        assert!(result.errors[0].contains(critical), "{:?}", result.errors);
    }
}

fn parsed_args(input: &str) -> serde_json::Value {
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1, "expected one call from {input:?}");
    serde_json::from_str(&calls[0].function.arguments).unwrap()
}

#[test]
fn qwen_attribute_drift_is_recovered_narrowly() {
    let args = parsed_args(
        "<function=get_weather>\n<parameter city=\"Santiago\">Chile</parameter>\n</function>",
    );
    assert_eq!(args["city"], "Santiago");

    let args = parsed_args(
        "<function=get_weather>\n<parameter name=\"city\">Santiago</parameter>\n</function>",
    );
    assert_eq!(args["city"], "Santiago");
}

#[test]
fn strict_parameter_wins_over_attribute_salvage() {
    let args = parsed_args(
        "<function=get_weather>\n<parameter=city>Kyoto</parameter>\n\
         <parameter city=\"Osaka\">x</parameter>\n</function>",
    );
    assert_eq!(args["city"], "Kyoto");
}

#[test]
fn attribute_salvage_rejects_ambiguous_shapes() {
    for body in [
        "<parameter city=Santiago>x</parameter>",
        "<parameter city=\"Santiago\" units=\"c\">x</parameter>",
        "<parameter 9city=\"Santiago\">x</parameter>",
    ] {
        let args = parsed_args(&format!("<function=get_weather>\n{body}\n</function>"));
        assert!(
            args.as_object().unwrap().is_empty(),
            "must not guess: {args}"
        );
    }
}

#[test]
fn attribute_salvage_does_not_cross_function_boundary() {
    let first = parse_qwen3_coder_call(
        "<function=get_weather>\n<parameter=city>Kyoto</parameter>\n</function>\n\
         <function=get_time>\n<parameter zone=\"JST\">x</parameter>\n</function>",
        0,
    )
    .unwrap();
    let args: serde_json::Value = serde_json::from_str(&first.function.arguments).unwrap();
    assert_eq!(args["city"], "Kyoto");
    assert!(args.get("zone").is_none());
}

#[test]
fn missing_gt_before_first_parameter_does_not_drop_parameter() {
    let parsed = parse_qwen3_coder_call(
        "<function=bash<parameter=command>ls -la</parameter></function>",
        0,
    )
    .expect("should recover function call");
    assert_eq!(parsed.function.name, "bash");
    let args: serde_json::Value = serde_json::from_str(&parsed.function.arguments).unwrap();
    assert_eq!(args["command"], "ls -la");
}

#[test]
fn coercion_handles_root_ref_nullable_and_type_arrays() {
    let emit_tree = ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "emit_tree".into(),
            description: None,
            parameters: Some(serde_json::json!({
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {
                            "val": {"type": "integer"},
                            "left": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]},
                            "right": {"type": ["object", "null"]},
                            "note": {"type": "string", "nullable": true},
                            "label": {"type": "string"}
                        },
                        "required": ["val", "left", "right"]
                    }
                },
                "$ref": "#/$defs/Node"
            })),
        },
    };
    let mut calls = vec![call(
        "emit_tree",
        r#"{"val":" 7 ","left":"null","right":"null","note":"null","label":"null"}"#,
    )];
    let changed = backfill_required_params(&mut calls, std::slice::from_ref(&emit_tree));
    assert_eq!(changed, 1);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["val"], 7);
    assert!(
        args["left"].is_null(),
        "anyOf null must coerce to null: {args}"
    );
    assert!(
        args["right"].is_null(),
        "type array null must coerce to null: {args}"
    );
    assert!(
        args["note"].is_null(),
        "OpenAPI nullable must coerce to null: {args}"
    );
    assert_eq!(
        args["label"], "null",
        "non-nullable string must keep literal 'null' as string"
    );
    assert!(
        find_empty_required_params(&calls[0], std::slice::from_ref(&emit_tree)).is_empty(),
        "nullable required params with JSON null must not be flagged empty"
    );
}

#[test]
fn blocking_parse_multiple_functions_and_invokes_in_single_envelope() {
    let qwen_multi = "<tool_call>\n\
        <function=read><parameter=path>a.rs</parameter></function>\n\
        <function=write><parameter=path>b.rs</parameter><parameter=content>hi</parameter></function>\n\
        </tool_call>";
    let (_, calls) = parse_tool_calls(qwen_multi);
    assert_eq!(calls.len(), 2, "expected both functions from one envelope");
    assert_eq!(calls[0].function.name, "read");
    assert_eq!(calls[1].function.name, "write");

    let mm_multi = "<minimax:tool_call>\n\
        <invoke name=\"read\"><parameter name=\"path\">a.rs</parameter></invoke>\n\
        <invoke name=\"bash\"><parameter name=\"command\">ls</parameter></invoke>\n\
        </minimax:tool_call>";
    let (_, mm_calls) = parse_tool_calls(mm_multi);
    assert_eq!(mm_calls.len(), 2, "expected both invokes from one envelope");
    assert_eq!(mm_calls[0].function.name, "read");
    assert_eq!(mm_calls[1].function.name, "bash");

    // Unclosed/truncated multi-function envelope recovers both blocks.
    let truncated = "<tool_call>\n\
        <function=read><parameter=path>a.rs</parameter></function>\n\
        <function=read><parameter=path>b.rs</parameter>";
    let (_, trunc_calls) = parse_tool_calls(truncated);
    assert_eq!(trunc_calls.len(), 2);
}

#[test]
fn truncated_trailing_parameter_tag_preserves_earlier_params() {
    let tc = parse_qwen3_coder_call(
        "<function=write><parameter=filePath>/tmp/a.rs</parameter><parameter=content",
        0,
    )
    .expect("earlier valid parameter must not be dropped by truncated trailing tag");
    assert_eq!(tc.function.name, "write");
    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
    assert_eq!(args["filePath"], "/tmp/a.rs");
    assert!(args.get("content").is_none());
}

#[test]
fn attribute_parameter_snippet_inside_code_does_not_inject_bogus_keys() {
    let tc = parse_qwen3_coder_call(
        "<function=write><parameter=code>\
         <parameter name=\"injected\">evil</parameter>\
         </parameter></function>",
        0,
    )
    .expect("should parse strict parameter");
    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
    assert_eq!(args.as_object().unwrap().len(), 1);
    assert!(args.get("injected").is_none());
    assert!(
        args["code"]
            .as_str()
            .unwrap()
            .contains("<parameter name=\"injected\">")
    );
}

#[test]
fn object_and_array_coercion_rejects_mismatched_json_kinds() {
    let def = ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "configure".into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "obj": {"type": "object"},
                    "arr": {"type": "array"}
                }
            })),
        },
    };
    let mut wrong = vec![call("configure", r#"{"obj":"[1,2]","arr":"{\"a\":1}"}"#)];
    assert_eq!(
        backfill_required_params(&mut wrong, std::slice::from_ref(&def)),
        0,
        "mismatched JSON kind must not be coerced"
    );
    let wrong_args: serde_json::Value = serde_json::from_str(&wrong[0].function.arguments).unwrap();
    assert!(wrong_args["obj"].is_string());
    assert!(wrong_args["arr"].is_string());

    let mut right = vec![call("configure", r#"{"obj":"{\"a\":1}","arr":"[1,2]"}"#)];
    assert_eq!(
        backfill_required_params(&mut right, std::slice::from_ref(&def)),
        1
    );
    let right_args: serde_json::Value = serde_json::from_str(&right[0].function.arguments).unwrap();
    assert!(right_args["obj"].is_object());
    assert!(right_args["arr"].is_array());
}

#[test]
fn non_command_tool_does_not_get_fabricated_description() {
    let ticket = ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "create_ticket".into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["title", "description"]
            })),
        },
    };
    let mut calls = vec![call("create_ticket", r#"{"title":"Login broken"}"#)];
    assert_eq!(
        backfill_required_params(&mut calls, std::slice::from_ref(&ticket)),
        0
    );
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert!(
        args.get("description").is_none(),
        "must not fabricate description for non-command tool: {args}"
    );
}

#[test]
fn coercion_normalizes_whole_floats_to_int_and_primitives_to_string() {
    let def = ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "record".into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "neg": {"type": "integer"},
                    "frac": {"type": "integer"},
                    "code": {"type": "string"},
                    "flag": {"type": "string"},
                    "mixed": {"type": ["string", "number"]}
                }
            })),
        },
    };
    let mut calls = vec![call(
        "record",
        r#"{"count":478.0,"neg":-12.0,"frac":3.5,"code":90210,"flag":true,"mixed":42}"#,
    )];
    assert_eq!(
        backfill_required_params(&mut calls, std::slice::from_ref(&def)),
        1
    );
    assert!(
        calls[0].function.arguments.contains(r#""count":478,"#),
        "whole float must serialize as exact integer: {}",
        calls[0].function.arguments
    );
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["count"].as_u64(), Some(478));
    assert_eq!(args["neg"].as_i64(), Some(-12));
    assert_eq!(args["frac"].as_f64(), Some(3.5));
    assert_eq!(args["code"], "90210");
    assert_eq!(args["flag"], "true");
    assert_eq!(args["mixed"], 42);
    assert_eq!(
        backfill_required_params(&mut calls, std::slice::from_ref(&def)),
        0,
        "coercion must be idempotent"
    );
}

#[test]
fn gemma4_to_json_escapes_quotes_newlines_tabs_and_backslashes() {
    let native = "{code:<|\"|>let s = \"hi\";\n\tlet p = \"C:\\tmp\r\";<|\"|>,n:42,ok:true}";
    let converted = gemma4_to_json(native);
    let parsed: serde_json::Value =
        serde_json::from_str(&converted).expect("escaped Gemma-4 JSON must parse cleanly");
    assert_eq!(parsed["code"], "let s = \"hi\";\n\tlet p = \"C:\\tmp\r\";");
    assert_eq!(parsed["n"], 42);
    assert_eq!(parsed["ok"], true);
}

#[test]
fn blocking_gemma4_with_json_args_does_not_leak_delimiters() {
    let input = "Let me check.<|tool_call>call:get_weather{\"city\": \"Paris\"}<tool_call|>";
    let (content, calls) = parse_tool_calls(input);
    assert_eq!(content.as_deref(), Some("Let me check."));
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Paris");

    let bare = "<|tool_call>call:get_weather{\"city\": \"Tokyo\"}<tool_call|>";
    let (bare_content, bare_calls) = parse_tool_calls(bare);
    assert!(bare_content.is_none(), "no delimiter leak into content");
    assert_eq!(bare_calls.len(), 1);
    assert_eq!(bare_calls[0].function.name, "get_weather");
}
