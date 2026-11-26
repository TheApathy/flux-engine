// SPDX-License-Identifier: AGPL-3.0-only

use serde_json::json;

use super::{
    MAX_SAFE_INTEGER, MAX_SCHEMA_DEPTH, MIN_SAFE_INTEGER, SchemaErrorKind, check_raw_schema_bytes,
    precheck_raw_chat_body, validate_strict_json_schema, validate_strict_json_schema_str,
};

#[test]
fn valid_openai_strict_schema_passes() {
    let schema = json!({
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 500
                        },
                        "score": {
                            "type": "integer",
                            "minimum": -100,
                            "maximum": 100
                        }
                    },
                    "required": ["explanation", "score"],
                    "additionalProperties": false
                }
            },
            "final_answer": { "type": ["string", "null"] }
        },
        "required": ["steps", "final_answer"],
        "additionalProperties": false
    });
    assert_eq!(validate_strict_json_schema(&schema), Ok(()));
}

#[test]
fn depth_at_limit_passes_and_exceeding_32_fails_with_exact_pointer() {
    let mut exact = json!({"type": "string"});
    for _ in 0..MAX_SCHEMA_DEPTH {
        exact = json!({
            "type": "object",
            "properties": { "child": exact },
            "required": ["child"],
            "additionalProperties": false
        });
    }
    assert_eq!(validate_strict_json_schema(&exact), Ok(()));

    let too_deep = json!({
        "type": "object",
        "properties": { "root": exact },
        "required": ["root"],
        "additionalProperties": false
    });
    let err = validate_strict_json_schema(&too_deep).expect_err("depth 33 must fail");
    assert_eq!(err.kind, SchemaErrorKind::MaxDepthExceeded);
    assert!(
        err.pointer
            .starts_with("#/properties/root/properties/child")
    );
    assert_eq!(
        err.pointer.matches("/properties/").count(),
        MAX_SCHEMA_DEPTH + 1
    );
}

#[test]
fn exact_ieee754_integer_boundaries_are_enforced() {
    let valid_edges = json!({
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "minimum": MIN_SAFE_INTEGER,
                "maximum": MAX_SAFE_INTEGER
            }
        }
    });
    assert_eq!(validate_strict_json_schema(&valid_edges), Ok(()));

    let above = json!({
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "maximum": MAX_SAFE_INTEGER + 1
            }
        }
    });
    let err = validate_strict_json_schema(&above).expect_err("2^53 must be rejected");
    assert_eq!(err.kind, SchemaErrorKind::UnsafeIntegerBound);
    assert_eq!(err.pointer, "#/properties/id/maximum");

    let below = json!({
        "type": "object",
        "properties": {
            "offset": {
                "type": "integer",
                "minimum": MIN_SAFE_INTEGER - 1
            }
        }
    });
    let err = validate_strict_json_schema(&below).expect_err("-(2^53) must be rejected");
    assert_eq!(err.kind, SchemaErrorKind::UnsafeIntegerBound);
    assert_eq!(err.pointer, "#/properties/offset/minimum");

    // Raw integer beyond u64::MAX (which serde_json would otherwise parse as f64).
    let raw = r#"{"type":"object","properties":{"big":{"type":"integer","maximum":99999999999999999999999999}}}"#;
    let err = validate_strict_json_schema_str(raw).expect_err("huge integer literal must fail");
    assert_eq!(err.kind, SchemaErrorKind::UnsafeIntegerBound);
    assert_eq!(err.pointer, "#/properties/big/maximum");
}

#[test]
fn lone_utf16_surrogates_fail_closed_with_exact_pointer() {
    // Valid surrogate pair (U+1F600 😀 = \uD83D\uDE00) passes!
    let valid_pair = r#"{"type":"object","properties":{"emoji":{"type":"string","description":"smile \uD83D\uDE00"}}}"#;
    assert_eq!(validate_strict_json_schema_str(valid_pair), Ok(()));

    // Lone high surrogate \uD800
    let lone_high = r#"{"type":"object","properties":{"bad":{"type":"string","description":"oops \uD800 end"}}}"#;
    let err = validate_strict_json_schema_str(lone_high).expect_err("lone high surrogate");
    assert_eq!(err.kind, SchemaErrorKind::LoneUtf16Surrogate);
    assert_eq!(err.pointer, "#/properties/bad/description");

    // Lone low surrogate \uDC00
    let lone_low = r#"{"type":"object","properties":{"bad":{"enum":["ok","\uDC00"]}}}"#;
    let err = validate_strict_json_schema_str(lone_low).expect_err("lone low surrogate");
    assert_eq!(err.kind, SchemaErrorKind::LoneUtf16Surrogate);
    assert_eq!(err.pointer, "#/properties/bad/enum/1");

    // Raw chat body precheck locates schema pointer before serde_json loses path context.
    let chat_body = br#"{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"test","strict":true,"schema":{"type":"object","properties":{"foo":{"description":"\uD800"}}}}}}"#;
    let err = precheck_raw_chat_body(chat_body).expect_err("chat precheck must reject surrogate");
    assert_eq!(err.kind, SchemaErrorKind::LoneUtf16Surrogate);
    assert_eq!(err.pointer, "#/properties/foo/description");

    // When strict is explicitly false, precheck defers to normal parsing.
    let non_strict = br#"{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"test","strict":false,"schema":{"type":"object"}}}}"#;
    assert_eq!(precheck_raw_chat_body(non_strict), Ok(()));
}

#[test]
fn contradictory_bounds_fail_closed_with_exact_pointer() {
    let min_gt_max = json!({
        "type": "object",
        "properties": {
            "age": { "type": "number", "minimum": 50, "maximum": 10 }
        }
    });
    let err = validate_strict_json_schema(&min_gt_max).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::ContradictoryBounds);
    assert_eq!(err.pointer, "#/properties/age/minimum");

    let len_inverted = json!({
        "type": "object",
        "properties": {
            "code": { "type": "string", "minLength": 12, "maxLength": 4 }
        }
    });
    let err = validate_strict_json_schema(&len_inverted).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::ContradictoryBounds);
    assert_eq!(err.pointer, "#/properties/code/minLength");

    let items_inverted = json!({
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": { "type": "string" },
                "minItems": 5,
                "maxItems": 2
            }
        }
    });
    let err = validate_strict_json_schema(&items_inverted).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::ContradictoryBounds);
    assert_eq!(err.pointer, "#/properties/tags/minItems");

    // RFC 6901 escaping (`/` -> `~1`, `~` -> `~0`) in property names.
    let escaped_key = json!({
        "type": "object",
        "properties": {
            "a/b~c": { "type": "integer", "minimum": 10, "maximum": 5 }
        }
    });
    let err = validate_strict_json_schema(&escaped_key).unwrap_err();
    assert_eq!(err.pointer, "#/properties/a~1b~0c/minimum");
}

#[test]
fn raw_scanner_rejects_pathological_deep_nesting_without_stack_overflow() {
    let mut raw = b"{\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"strict\":true,\"schema\":".to_vec();
    raw.extend(std::iter::repeat_n(b'[', 5_000));
    raw.extend(std::iter::repeat_n(b']', 5_000));
    raw.extend_from_slice(b"}}}");
    let err = precheck_raw_chat_body(&raw).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::MaxDepthExceeded);
}

#[test]
fn invalid_type_keywords_fail_closed_with_exact_pointer() {
    let bad_root = json!({ "type": "definitely_not_a_valid_json_type" });
    let err = validate_strict_json_schema(&bad_root).expect_err("unknown root type must fail");
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchemaType);
    assert_eq!(err.pointer, "#/type");

    let bad_nested = json!({
        "type": "object",
        "properties": {
            "count": { "type": ["integer", "bogus"] }
        }
    });
    let err = validate_strict_json_schema(&bad_nested).expect_err("unknown union type must fail");
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchemaType);
    assert_eq!(err.pointer, "#/properties/count/type/1");

    let empty_union = json!({ "type": [] });
    let err = validate_strict_json_schema(&empty_union).expect_err("empty type array must fail");
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchemaType);
    assert_eq!(err.pointer, "#/type");

    let dup_union = json!({ "type": ["string", "string"] });
    let err = validate_strict_json_schema(&dup_union).expect_err("duplicate type union must fail");
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchemaType);
    assert_eq!(err.pointer, "#/type/1");

    let valid_union = json!({ "type": ["string", "null"] });
    assert_eq!(validate_strict_json_schema(&valid_union), Ok(()));
}

#[test]
fn raw_scanner_preserves_non_ascii_utf8_json_pointers() {
    let raw = r#"{"type":"object","properties":{"café":{"const":"\uD800"}}}"#;
    let err = check_raw_schema_bytes(raw.as_bytes()).expect_err("lone surrogate must fail");
    assert_eq!(err.kind, SchemaErrorKind::LoneUtf16Surrogate);
    assert_eq!(err.pointer, "#/properties/café/const");
}
